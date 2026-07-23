# Data Quality

A backtester's credibility depends as much on data quality as on strategy
logic — a strategy validated against stale or silently-adjusted prices
proves nothing. This module makes explicit what a data provider does and
doesn't guarantee, and flags likely data problems in what's already been
fetched.

**Scope, stated explicitly upfront:** this is metadata and heuristic
detection on top of `yfinance` — not a new, more reliable data source.
Building a real point-in-time / survivorship-free provider (no silent
historical revisions, delisted securities remain queryable) needs a paid
data vendor (Polygon, Norgate, Sharadar, etc.) this library has no
credentials for — `DataFactory`'s `alpaca`/`polygon`/`bloomberg` provider
names are still `NotImplementedError` placeholders, a real and
already-documented blocker, not something this module works around.

---

## Dataset Metadata (`data/metadata.py`)

Every `DataProvider` implements `get_metadata(symbol, interval="1d") ->
DataSetMetadata`, an **honest self-report** — not an aspirational one.

```python
from standard_quant_tools.data.factory import DataFactory

provider = DataFactory.get_provider()
meta = provider.get_metadata("AAPL")
print(meta)
# DataSetMetadata(provider='yfinance', adjusted=True, survivorship_free=False,
#                  point_in_time=False, frequency='1d', timezone='America/New_York',
#                  retrieved_at='2026-07-23T...')
```

| Field | YFinanceProvider value | Why |
|---|---|---|
| `adjusted` | `True` | yfinance auto-adjusts for splits/dividends by default |
| `survivorship_free` | `False` | Not a yfinance guarantee — delisted tickers may become unqueryable |
| `point_in_time` | `False` | Not a yfinance guarantee — historical values can be silently revised |
| `frequency` | echoes the requested `interval` | — |
| `timezone` | Inferred from the symbol's Yahoo Finance exchange suffix via a ~19-entry lookup table (`_EXCHANGE_SUFFIX_TIMEZONES`), e.g. `.L`→`Europe/London`, `.DE`→`Europe/Berlin`, `.HK`→`Asia/Hong_Kong`; any symbol whose suffix isn't in that table — including all unsuffixed US tickers — defaults to `"America/New_York"` | Local, no-network heuristic based on ticker convention, not a provider-verified exchange timezone — yfinance doesn't expose a reliable per-symbol timezone through this provider's interface |
| `retrieved_at` | current UTC timestamp | When this metadata object was generated, not when the underlying data was last updated upstream |

A provider that could make stronger guarantees (a real point-in-time
vendor) would report `True` for the relevant fields — the model exists
precisely so that claim becomes visible and checkable, not implicit.

---

## Data Quality Checks (`data/quality.py`)

Three pure functions operating on an already-fetched OHLCV `DataFrame` —
no new data source, no network calls.

```python
from standard_quant_tools.data.quality import (
    detect_missing_bars, detect_stale_prices, detect_price_jumps,
)

df = provider.get_ohlcv("AAPL", "2023-01-01", "2024-01-01")

gaps = detect_missing_bars(df)
stale = detect_stale_prices(df, n=3)
jumps = detect_price_jumps(df, threshold=0.15)
```

**`detect_missing_bars(df)`** — flags weekday gaps in the index. **Calendar-free
heuristic, stated explicitly:** it infers expected trading days from the
data's own weekday pattern (`pandas.bdate_range`), not a real market-holiday
calendar (this repo doesn't depend on `pandas_market_calendars` or similar
— see [04_backtesting.md](04_backtesting.md)'s minimal-dependency stance).
U.S. market holidays (Thanksgiving, Christmas, etc.) will therefore show up
as false-positive "gaps." Treat findings as leads to investigate, not
proven defects.

**`detect_stale_prices(df, n=3)`** — flags runs of `n`+ consecutive
identical `Close` values, a likely stale/frozen quote (a real market rarely
closes at the exact same price for multiple consecutive sessions).

**`detect_price_jumps(df, threshold=0.15)`** — flags single-bar
Close-to-Close moves exceeding `threshold`, a proxy for an unadjusted
split/dividend or a data error. A genuinely volatile session produces the
same signature, so this is a lead, not a proven defect either.

---

## Agent Tool: `get_data_quality_report`

Combines both pieces above into one JSON-shaped call for LLM tool-calling.

```python
from standard_quant_tools.agent.tools import get_data_quality_report
from standard_quant_tools.agent.models import DataQualityReportInput

result = get_data_quality_report(DataQualityReportInput(
    symbol="AAPL", start_date="2023-01-01", end_date="2024-01-01",
    stale_run_length=3, jump_threshold=0.15,
))

print(result.metadata)          # dataset provenance, as a dict
print(result.missing_bars)      # [{"date": ..., "weekday": ...}, ...]
print(result.stale_price_runs)  # [{"start": ..., "end": ..., "price": ..., "run_length": ...}, ...]
print(result.price_jumps)       # [{"date": ..., "pct_change": ...}, ...]
```

See [09_advanced_agent_tools.md](09_advanced_agent_tools.md) for the tool's
full input/output reference alongside the rest of the agent tools.
