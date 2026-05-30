# Stock Screener

The screener evaluates a list of tickers concurrently against fundamental and technical filters, returning a sorted `pd.DataFrame` of passing stocks.

**Small universes (≤ 20 tickers):** all network calls run in parallel via `asyncio.gather` — screening 20 tickers takes roughly the same wall time as screening 5.

**Large universes (> 20 tickers):** the ticker list is automatically split across multiple `ProcessPoolExecutor` workers. Each worker runs its own asyncio event loop, bypassing the GIL for the full pipeline (fetch + indicator compute). Combined with the Parquet disk cache, repeated runs on the same universe are near-instant.

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
        "price_above_sma": 50,   # price above 50-day SMA
        "price_above_sma": 200,  # above 200-day SMA (golden zone)
    },
    sort_by="rsi_14",
    ascending=False,  # highest RSI first
)
```

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
| `None` (default) | Auto: 1 for ≤ 20 tickers, `cpu_count` for larger universes |
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
