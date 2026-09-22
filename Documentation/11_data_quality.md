# Data Quality

A backtester's credibility depends as much on data quality as on strategy
logic — a strategy validated against stale or silently-adjusted prices
proves nothing. This module makes explicit what a data provider does and
doesn't guarantee, and flags likely data problems in what's already been
fetched.

**Scope, stated explicitly upfront:** this is metadata and heuristic
detection layered on top of whichever provider is actually configured —
not a new, more reliable data source in itself. `DataFactory` has four real
implementations — `YFinanceProvider`, `PolygonProvider`,
`BloombergProvider`, `DatabentoProvider` (only `alpaca` is still a
`NotImplementedError` placeholder) — but integrating a provider is not the
same thing as it being point-in-time or survivorship-free. **Every
provider's `get_metadata()` reports `point_in_time=False`**, Databento
included: it announces its reprocessing rather than restating silently,
which is better than most and is still not the guarantee the field asks
about. `survivorship_free` is `False` on three of the four; Databento
reports `True`, because its archive is organized by publication, so an
instrument that stopped trading stays queryable over the window it traded
in.

Point-in-time *fundamentals* are a separate contract from that flag and
they do exist: `get_point_in_time_records` serves one row per version of a
fact with `event_time` and `available_time`, from Polygon only, and every
other provider refuses by name — see
[01_data_fetching.md](01_data_fetching.md#point-in-time-records-get_point_in_time_records).
A point-in-time *price* feed (Norgate, Sharadar, a point-in-time-specific
Bloomberg endpoint this library doesn't use) is still a real gap, not
something this module works around.

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
| `timezone` | Inferred from the symbol's Yahoo Finance exchange suffix via a ~19-entry lookup table (`_EXCHANGE_SUFFIX_TIMEZONES`), e.g. `.L`→`Europe/London`, `.DE`→`Europe/Berlin`, `.HK`→`Asia/Hong_Kong`; any symbol whose suffix isn't in that table — including all unsuffixed US tickers — defaults to `"America/New_York"` | A LABEL for an already-normalised index, not an instruction: daily bars are naive session dates in this zone, intraday bars naive UTC instants, so localising to it before a join aligns nothing. Local, no-network heuristic based on ticker convention, not a provider-verified exchange timezone |
| `notes` | free prose, usually empty here | What the four booleans cannot say — which feed answered, what the index is, what the provider refuses. This is where Databento names its sample feed, and no flag could have |
| `retrieved_at` | current UTC timestamp | When this metadata object was generated, not when the underlying data was last updated upstream |

A provider that could make stronger guarantees (a real point-in-time
vendor) would report `True` for the relevant fields — the model exists
precisely so that claim becomes visible and checkable, not implicit.

**The one value here that surprises people is Databento's
`adjusted=False`.** It serves what the venue published, so a split is a
real -50% bar; every other provider reports `True`. Read it before running
`detect_price_jumps` over a Databento frame, where an unadjusted corporate
action is a finding rather than a false positive.

---

## Data Quality Checks (`data/quality.py`)

Four pure functions operating on an already-fetched OHLCV `DataFrame` —
no new data source, no network calls.

```python
from standard_quant_tools.data.quality import (
    detect_missing_bars, detect_volume_anomalies,
    detect_stale_prices, detect_price_jumps,
)

df = provider.get_ohlcv("AAPL", "2023-01-01", "2024-01-01")

gaps = detect_missing_bars(df, calendar="XNYS")
thin = detect_volume_anomalies(df, window=20, thin_fraction=0.05)
stale = detect_stale_prices(df, n=3)
jumps = detect_price_jumps(df, threshold=0.15)
```

**`detect_missing_bars(df, calendar="XNYS")`** — flags sessions missing from
the index. **Against the exchange calendar when it can:** with the optional
`exchange_calendars` package installed, expected sessions come from the named
calendar and a holiday is not a gap. Without it, the function falls back to
the data's own weekday pattern (`pandas.bdate_range`), and U.S. market
holidays (Thanksgiving, Christmas, etc.) show up as false-positive "gaps" —
on a live year every one of the 21 reported gaps was a holiday, which is why
the calendar path exists. Each entry says which it used, `"basis":
"calendar"` or `"weekday"`, so a reader knows whether a finding is a lead or
a defect.

**`detect_volume_anomalies(df, window=20, thin_fraction=0.05)`** — flags bars
whose `Volume` is zero (`kind="zero"`) or below `thin_fraction` of the
trailing `window`-bar median (`kind="thin"`), with the median beside each.
A sample feed that carries a few percent of the consolidated tape reads as
thin against a full-volume history, and a halted session reads as zero;
neither is visible from prices alone.

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
It lives in the `research` runtime, not in `data` — a second name for these
checks beside the fetch tools is the duplication the runtime split exists to
prevent.

Four arguments decide what it measures. `source` names any registered
provider, so a single-venue tape or a vendor extract — the feeds most worth
checking — can be checked at all; an unknown name is refused and the legal
ones named. `calendar` is the exchange code the gap detector expects
sessions from: the same 2024 US equity frame reports no gaps under `XNYS`
and nine under `XTKS`, and a code the library does not recognize falls back
to weekdays with every entry carrying `basis="weekday"`. `volume_window` and
`thin_fraction` are the volume-anomaly knobs; the default fraction is
deliberately severe, so a feed carrying a few percent of consolidated volume
all the way through looks normal — it is thin CONSISTENTLY, not
occasionally. Raise it (0.5 with a short window) to ask whether volume is
thin relative to its own recent past.

```python
from standard_quant_tools.agent.tools import get_data_quality_report
from standard_quant_tools.agent.models import DataQualityReportInput

result = get_data_quality_report(DataQualityReportInput(
    symbol="AAPL", start_date="2023-01-01", end_date="2024-01-01",
    source="databento", calendar="XNYS",
    stale_run_length=3, jump_threshold=0.15,
    volume_window=20, thin_fraction=0.05,
))

print(result.metadata)          # dataset provenance, as a dict
print(result.missing_bars)      # [{"date": ..., "weekday": ..., "basis": "calendar" | "weekday"}, ...]
print(result.stale_price_runs)  # [{"start": ..., "end": ..., "price": ..., "run_length": ...}, ...]
print(result.price_jumps)       # [{"date": ..., "pct_change": ...}, ...]
print(result.volume_anomalies)  # [{"date": ..., "volume": ..., "trailing_median": ..., "kind": "zero" | "thin"}, ...]
```

See [09_advanced_agent_tools.md](09_advanced_agent_tools.md) for the tool's
full input/output reference alongside the rest of the agent tools.
