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
action is a finding rather than a false positive. The dataset build reads
it: see [15_modeling.md, *Splits in unadjusted bars*](15_modeling.md#splits-in-unadjusted-bars).

---

## What the provider drops and flags before any check runs

Two conditions are handled by every provider before a frame reaches a
check or a tool, and both are written on `df.attrs` (see
[01_data_fetching.md](01_data_fetching.md#a-bar-with-no-close-is-dropped-and-a-bar-still-trading-is-flagged)):

- **A bar with no Close is dropped.** After the last priced bar it is a
  placeholder for a session that has not traded (`dropped_placeholder_bars`);
  before it, a missing bar (`dropped_missing_bars`). A whole window without
  a Close is refused with `NonRetryableAPIError`, once. A dropped interior
  bar is therefore a gap in the index, and `detect_missing_bars` reports it
  like any other gap — against the calendar, so it is a finding rather than
  a holiday.
- **A last daily bar whose session has not closed is flagged**
  (`partial_last_bar`, with the session's date and the UTC instant it
  closes), judged against the exchange calendar when `exchange_calendars` is
  installed and the venue's regular close otherwise. Its volume is a
  partial session's, so `detect_volume_anomalies` may report it as thin —
  the flag says why.

The agent tools that fetch bars say both in `warnings`, per symbol.

---

## Data Quality Checks (`data/quality.py`)

Pure functions operating on an already-fetched OHLCV `DataFrame` — no new
data source, no network calls. Four are heuristics whose findings are leads
(gaps, thin volume, stale prices, jumps); three are integrity checks whose
findings are data errors (duplicate labels, labels out of order, bars whose
prices contradict each other); one reads provenance rather than numbers
(a sample feed).

```python
from standard_quant_tools.data.quality import (
    detect_missing_bars, detect_volume_anomalies,
    detect_stale_prices, detect_price_jumps,
    detect_duplicate_timestamps, detect_out_of_order_timestamps,
    detect_ohlc_inconsistencies, detect_sample_feed,
)

df = provider.get_ohlcv("AAPL", "2023-01-01", "2024-01-01")

gaps = detect_missing_bars(df, calendar="XNYS")
thin = detect_volume_anomalies(df, window=20, thin_fraction=0.05)
stale = detect_stale_prices(df, n=3)
jumps = detect_price_jumps(df, threshold=0.15)
repeated = detect_duplicate_timestamps(df)
backwards = detect_out_of_order_timestamps(df)
contradictory = detect_ohlc_inconsistencies(df)
sample = detect_sample_feed(df)     # None unless attrs["dataset"] names a sample feed
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
a defect. The span checked runs from the earliest bar to the latest, so an
index out of order does not shrink it.

**`detect_volume_anomalies(df, window=20, thin_fraction=0.05)`** — flags bars
whose `Volume` is zero (`kind="zero"`) or below `thin_fraction` of the
trailing `window`-bar median (`kind="thin"`), with the median beside each.
A halted session reads as zero and a bar thin next to its neighbours reads
as thin; neither is visible from prices alone. **It cannot find a sample
feed.** Each bar is judged against the frame's own history, so the test is
blind to scale: multiplying every volume by 0.036 gives the same answer, and
a feed carrying 3.6% of the tape on every bar reads exactly like the tape.
Only a frame that switches feeds part-way shows a thin stretch.

**`detect_sample_feed(df)`** — answers the question no statistic can, from
provenance. A provider that chooses among datasets stamps the one that
answered on the frame (`df.attrs["dataset"]`; Databento does, including on a
cache hit), and this returns `{"dataset", "provider", "note"}` when that
dataset is a known sample of the tape — `EQUS.MINI`, whose volume is 2-4% of
consolidated and whose daily close is the last print of the UTC day. `None`
means "not known to be a sample", not "known to be the tape": a frame with no
stamp says nothing either way.

**`detect_duplicate_timestamps(df)`** — bar labels that occur more than
once, each with its count and row positions. Two rows under one timestamp
are two answers to one question: a join or `.loc` lookup returns both, a
resample counts the bar twice.

**`detect_out_of_order_timestamps(df)`** — rows whose label is earlier than
the row before it, with both labels. Every rolling window, return and fill
reads the rows in order as time, so a swapped pair corrupts each of them
without an error. A repeated label is a duplicate, not out of order.

**`detect_ohlc_inconsistencies(df)`** — bars whose prices contradict each
other: `Low` above `High` (`kind="low_above_high"`, reported once, since the
range is then empty), or `Open`/`Close` outside `[Low, High]`
(`"open_outside_range"`, `"close_outside_range"`). A bar's high and low bound
every trade in it, so any of these is a data error rather than a market
event, and every Close-only check above is blind to it. A relative tolerance
of `1e-9` absorbs floating-point noise in a scaled or adjusted price, so a
bar with all four prices equal is never flagged.

**`detect_stale_prices(df, n=3)`** — flags runs of `n`+ consecutive
identical `Close` values, a likely stale/frozen quote (a real market rarely
closes at the exact same price for multiple consecutive sessions).

**`detect_price_jumps(df, threshold=0.15)`** — flags single-bar
Close-to-Close moves exceeding `threshold`, a proxy for an unadjusted
split/dividend or a data error. A genuinely volatile session produces the
same signature, so this is a lead, not a proven defect either.

**`detect_split_like_moves(close, threshold=0.35)`** — the split screen the
backtest and the dataset build run, as records. It lists every
Close-to-Close move beyond `threshold` (default `SPLIT_SCREEN_THRESHOLD`,
in `standard_quant_tools.constants`), and every fall within 10% on a log
scale of a 3:2 split (26.3% to 39.7%) however far below `threshold`, with
its `date`, `close_move`, `split_ratio` and `ratio_error`. `split_ratio` is the listed ratio (3:2 to
50:1, or a reverse) that the price ratio across the bar is within 10% of on
a log scale, else `None`, in new shares per old share — the unit
`DatasetSpec.corporate_actions` takes. `ratio_error` is the log distance to
the nearest listed ratio. Consistency is not proof: a −90% day reads like a
10:1 split and a −30% day like a 3:2 split. 4:3 and 5:4 are not named below
the threshold: their bands reach down to falls of 17% and 12%, where
ordinary moves are common (on GARCH-t(4) series at 2–4% daily volatility a
4:3 band named an ordinary fall once per 14 to 2 name-years, against once
per 75 to 9 for 3:2).

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

Whether the bars came from a sample feed is answered by provenance instead:
`served_dataset` is the dataset the provider stamped on the frame (`None`
for a provider that names none), and `sample_feed` is `True` — with
`sample_feed_note` saying what the feed is — when that dataset is a known
sample of the tape. The three integrity checks run on every report and take
no arguments: `duplicate_timestamps`, `out_of_order_timestamps` and
`ohlc_inconsistencies` are data errors rather than leads, and nothing
downstream refuses them.

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
print(result.served_dataset, result.sample_feed)   # e.g. "EQUS.MINI", True
print(result.duplicate_timestamps)     # [{"timestamp": ..., "count": ..., "positions": [...]}, ...]
print(result.out_of_order_timestamps)  # [{"position": ..., "timestamp": ..., "previous": ...}, ...]
print(result.ohlc_inconsistencies)     # [{"date": ..., "position": ..., "kind": ..., "open": ..., "high": ..., "low": ..., "close": ...}, ...]
```

See [09_advanced_agent_tools.md](09_advanced_agent_tools.md) for the tool's
full input/output reference alongside the rest of the agent tools.
