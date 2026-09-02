# Stock Screener

The screener evaluates a list of tickers concurrently against fundamental and technical filters, returning a `pd.DataFrame` of passing stocks, optionally sorted by a chosen column.

**Small universes (≤ 20 tickers):** all network calls run in parallel via `asyncio.gather` — screening 20 tickers takes roughly the same wall time as screening 5.

**Large universes (> 20 tickers):** the ticker list is automatically split across multiple `ProcessPoolExecutor` workers. Each worker runs its own asyncio event loop, bypassing the GIL for the full pipeline (fetch + indicator compute). Combined with the Parquet disk cache, repeated runs on the same universe are near-instant.

**Beta filter optimisation:** When `beta_max` or `beta_min` filters are present, SPY OHLCV data is fetched **once per `screen_stocks_async()` invocation** and reused for every ticker in that invocation that needs a beta computation. For a single-process run (≤ 20 tickers, or `n_workers=1`) that means one SPY fetch total; when the universe is split across a `ProcessPoolExecutor`, each worker independently prefetches SPY once for its own batch, i.e. one fetch per worker. On a 500-ticker universe where 200 tickers require beta, screened with the default 8-worker split, this means 8 SPY fetches instead of up to 200 — eliminating roughly 192 redundant HTTP requests compared to the naïve per-ticker fetch. If the SPY prefetch itself fails, it is silently skipped and each ticker needing beta falls back to fetching SPY individually.

**Error handling:** a per-ticker failure (network error, missing data, bad ratio, indicator that can't be computed, etc.) is never indistinguishable from a ticker that simply failed a filter condition. `_fetch_ticker_data` returns a `(status, ticker, payload)` tuple — `"passed"`, `"failed_filter"`, or `"error"` — and both `screen_stocks_async` and `screen_stocks` surface the non-passing cases via `DataFrame.attrs` on every DataFrame they return (including the empty-result case):

| `attrs` key | Type | Meaning |
|---|---|---|
| `failed_filters` | `Dict[ticker, str]` | Genuine rejection — maps to the specific filter key the ticker failed, e.g. `{"AAPL": "pe_ratio_max"}` |
| `failed_tickers` | `Dict[ticker, str]` | Data-fetch/compute exception — maps to the exception's string message |
| `failed_batches` | `List[str]` | `screen_stocks` only. Error message per worker process that raised *before* returning any per-ticker result at all (`n_workers > 1` only — always `[]` for single-process runs) |

```python
result = screen_stocks(tickers, filters={"pe_ratio_max": 15})
print(result.attrs["failed_filters"])   # {"AAPL": "pe_ratio_max", ...}
print(result.attrs["failed_tickers"])   # {"XYZ": "HTTPError: 429 ...", ...}
print(result.attrs["failed_batches"])   # [] unless a whole worker process died
```

Unknown filter keys are rejected up front: `screen_stocks` / `screen_stocks_async` raise `ValidationError` before making any network call if `filters` contains a key outside the fixed set documented below.

**Filter values are validated too, not just their names.** A bound must be a finite number, and the two window filters (`price_above_sma`, `price_below_sma`) must be positive whole numbers. NaN is the case worth naming: NaN fails *every* comparison, so `rsi_max=float("nan")` made `last_rsi > rsi_max` False for every ticker and an oversold screen silently became a no-op that admitted RSI 100. A filter that rejects nothing is indistinguishable from a filter nothing failed. Wrong types and out-of-range windows used to raise inside the per-ticker `try`/`except`, so one malformed filter came back as *N* identical `"error"` entries across the universe with nothing saying the filter itself was the problem.

**A beta that cannot be estimated is an error, not a value.** `calculate_beta` returns `alpha`/`beta`/`r_squared` all **NaN** when fewer than two points overlap the benchmark. It USED TO return `0.0` for all three, which was indistinguishable from a real answer because 0.0 is also a legitimate beta — and the screener *filtered* on it, so a ticker whose history did not overlap SPY at all reported beta 0.0 and **passed** `beta_max=0.5`. "Could not be estimated" was read as "very low beta", which is backwards for the defensive screen that bound exists to express. The screener now requires a minimum overlap and reports a shortfall as a per-ticker error in `failed_tickers`.

That minimum is **configurable**, because 20 is a judgment call rather than a mathematical bound — screening weekly bars, or deliberately hunting recent listings, are legitimate reasons to lower it:

```python
screen_stocks(tickers, {"beta_max": 1.2}, min_beta_obs=10)   # default 20
```

Available as `min_beta_obs` on `screen_stocks`, `screen_stocks_async`, and `ScreenerInput` for the agent tool. `DEFAULT_MIN_BETA_OBS` is the exported default.

**The floor is bounded below at 2, and that bound is not a matter of taste.** `calculate_beta` returns its all-zero sentinel below two overlapping points, and that sentinel is indistinguishable from a real beta of `0.0` — so any floor under 2 would reopen precisely the bug the floor exists to close. Values below 2, non-integers, and `True` (which subclasses `int`) all raise `ValidationError` up front rather than once per ticker.

The floor is applied identically whether the run is single-process or split across a `ProcessPoolExecutor`. That is worth stating because it is exactly where such a parameter goes wrong: the worker rebuilds its call from a plain tuple, and a value left out of that tuple does not fail — it silently reverts to the default inside the child, so the same request would screen differently at `n_workers=1` than at `n_workers=8`. The worker unpacks strictly, so omitting a parameter is an immediate error rather than a quiet divergence.

**Multi-worker `attrs` merging:** `pd.concat` does not reliably propagate `.attrs` — pandas only keeps them when every concatenated frame's `.attrs` are identical, and drops them otherwise, so naively concatenating each worker's batch DataFrame would silently lose `failed_filters` / `failed_tickers` from all but (at best) one batch. `.attrs` itself *does* survive the trip through `ProcessPoolExecutor` (pandas includes `_attrs` in `__getstate__`/`__setstate__`, so pickling a DataFrame back from a worker via `future.result()` preserves it) — `screen_stocks` relies on that and works around the `pd.concat` limitation explicitly: it reads `.attrs["failed_filters"]` / `.attrs["failed_tickers"]` off each worker's batch DataFrame individually, merges them into two dicts in the parent process, then assigns the merged dicts onto the final concatenated DataFrame's `.attrs` after `pd.concat` runs — overwriting whatever (if anything) `pd.concat` produced on its own. So failure info from every worker batch is preserved, not just the last one.

---

## Basic Usage

```python
from standard_quant_tools.screener import screen_stocks

result = screen_stocks(
    tickers=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN"],
    filters={
        "pe_ratio_max": 35,
        "rsi_max": 55,
        "price_above_sma": 50,
    },
    sort_by="rsi_14",
    ascending=True,   # most oversold first
)
print(result)
```

---

## All Available Filters

### Fundamental Filters

| Key | Description | Example |
|---|---|---|
| `pe_ratio_max` | Forward P/E upper bound | `25` |
| `pb_ratio_max` | Price-to-Book upper bound | `5.0` |
| `debt_equity_max` | Debt-to-Equity upper bound | `150` |
| `roe_min` | Return on Equity lower bound (decimal) | `0.15` = 15% |
| `profit_margin_min` | Net profit margin lower bound (decimal) | `0.10` = 10% |
| `div_yield_min` | Dividend yield lower bound (decimal) | `0.02` = 2% |
| `market_cap_min` | Market cap lower bound (USD) | `10_000_000_000` = $10B |

### Technical Filters

| Key | Description | Example |
|---|---|---|
| `rsi_max` | RSI(14) must be below this | `40` (oversold screen) |
| `rsi_min` | RSI(14) must be above this | `60` (momentum screen) |
| `price_above_sma` | Close must be above SMA(N) | `50` = above 50-day SMA |
| `price_below_sma` | Close must be below SMA(N) | `200` = below 200-day SMA |
| `beta_max` | Beta vs SPY upper bound | `1.2` |
| `beta_min` | Beta vs SPY lower bound | `0.5` |

---

## Example Screens

### Value Screen

```python
result = screen_stocks(
    tickers=["AAPL", "MSFT", "GOOGL", "TSLA", "JPM", "BAC", "WMT", "KO"],
    filters={
        "pe_ratio_max": 20,
        "pb_ratio_max": 3.0,
        "debt_equity_max": 100,
        "roe_min": 0.15,
    },
    sort_by="forward_pe",
    ascending=True,
)
```

### Momentum Screen

```python
result = screen_stocks(
    tickers=["NVDA", "AMD", "AAPL", "MSFT", "TSLA", "META"],
    filters={
        "rsi_min": 60,           # strong momentum
        "price_above_sma": 200,  # above 200-day SMA (golden zone)
    },
    sort_by="rsi_14",
    ascending=False,  # highest RSI first
)
```

> `filters` accepts only one `price_above_sma` / `price_below_sma` value per call — it's a single SMA period, not a set. To require price above *both* the 50- and 200-day SMA, screen on one and check the other with a follow-up `get_technical_analysis` call.

### Oversold Quality Screen

```python
result = screen_stocks(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "V", "MA", "UNH"],
    filters={
        "rsi_max": 40,             # oversold
        "profit_margin_min": 0.15, # profitable company
        "beta_max": 1.5,           # not too volatile
        "market_cap_min": 50_000_000_000,  # large cap only
    },
    sort_by="rsi_14",
    ascending=True,  # most oversold first
    start_date="2023-06-01",
    end_date="2024-01-01",
)
print(result[['rsi_14', 'forward_pe', 'return_on_equity', 'beta']])
```

### Low-Beta Dividend Screen

```python
result = screen_stocks(
    tickers=["KO", "PEP", "JNJ", "PG", "MCD", "T", "VZ", "O"],
    filters={
        "div_yield_min": 0.025,   # at least 2.5% dividend yield
        "beta_max": 0.8,          # defensive
        "debt_equity_max": 200,
    },
    sort_by="dividend_yield",
    ascending=False,
)
```

---

## Large Universe Screening

For 100+ tickers, pass `n_workers` to control the process pool. Combined with the Parquet cache, the second run of the same universe is dramatically faster.

```python
# S&P 500 screen — first run fetches from yfinance, writes Parquet cache
# Subsequent runs read from disk (~10× faster per ticker)
sp500 = [...]  # your 500-ticker list

result = screen_stocks(
    tickers=sp500,
    filters={
        "pe_ratio_max": 25,
        "roe_min": 0.15,
        "rsi_max": 50,
        "market_cap_min": 10_000_000_000,
    },
    sort_by="rsi_14",
    ascending=True,
    n_workers=8,    # 8 parallel processes, each running asyncio.gather on their batch
)
print(f"Passed: {len(result)} / {len(sp500)}")
```

| `n_workers` | Behaviour |
|---|---|
| `None` (default) | Auto: 1 for ≤ 20 tickers; otherwise `min(cpu_count, max(n // 10, 2))` — approaches `cpu_count` as the universe grows, but is capped lower for universes just over the 20-ticker threshold |
| `1` | Single process (asyncio only) — best for small lists and notebooks |
| `> 1` | ProcessPoolExecutor — best for 50+ tickers |

---

## Via Agent Tool

```python
from standard_quant_tools.agent.tools import run_screener
from standard_quant_tools.agent.models import ScreenerInput

result = run_screener(ScreenerInput(
    tickers=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
    filters={"pe_ratio_max": 35, "rsi_max": 50, "beta_max": 1.5},
    sort_by="rsi_14",
    ascending=True,
))

print(f"Passed: {result.num_passed} / {5}")
print(f"Tickers: {result.tickers_passed}")
for row in result.results:
    print(row)

# Same failed_filters / failed_tickers / failed_batches breakdown as the
# DataFrame .attrs, but as plain Pydantic fields on the result:
print(result.failed_filters)   # {ticker: filter key it failed}
print(result.failed_tickers)   # {ticker: error message}
print(result.failed_batches)   # [error message, ...] (n_workers > 1 only)
```

The `ScreenerResult` Pydantic model is directly JSON-serializable for LLM consumption.

---

## Async Usage

For embedding in a larger async application:

```python
import asyncio
from standard_quant_tools.screener import screen_stocks_async

async def main():
    result = await screen_stocks_async(
        tickers=["AAPL", "MSFT", "GOOGL"],
        filters={"pe_ratio_max": 35, "rsi_max": 60},
        sort_by="rsi_14",
    )
    return result

df = asyncio.run(main())
```

`screen_stocks_async` sets `df.attrs["failed_filters"]` and `df.attrs["failed_tickers"]` on its result (see Error handling above), but never `failed_batches` — that key only exists on results from `screen_stocks`, since it's specific to the `ProcessPoolExecutor` batch-splitting path.

---

## Output Columns

The returned DataFrame always includes `ticker` as the index. Available columns depend on which filters were applied:

| Column | Present when |
|---|---|
| `forward_pe` | Any fundamental filter |
| `price_to_book` | Any fundamental filter |
| `debt_to_equity` | Any fundamental filter |
| `return_on_equity` | Any fundamental filter |
| `profit_margins` | Any fundamental filter |
| `dividend_yield` | Any fundamental filter |
| `market_cap` | Any fundamental filter |
| `last_close` | Any technical filter |
| `rsi_14` | `rsi_max` or `rsi_min` |
| `sma_{N}` | `price_above_sma` or `price_below_sma` |
| `beta` | `beta_max` or `beta_min` |
