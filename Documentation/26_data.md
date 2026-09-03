# The Data Runtime

Seventeen tools for getting the bytes and saying what they can support.

Every other runtime answers a question about markets. This one answers where
the data is — and it is the only runtime whose output is meant to be
consumed by all the others.

## Why this exists

Before it, the raw data layer was reachable only by going through an
analysis tool that wanted to do something else with the bars. Two agents
asking about the same universe fetched it twice. Worse, the frame each tool
built died inside the call that built it: a return panel, an OHLCV panel or
a tick tape could not be handed to the next runtime without being recomputed
from scratch.

That is the thing being fixed. Not a missing algorithm — a **computed
intermediate state that was thrown away instead of becoming something
another agent could reason over.**

## These tools return a reference, not the data

Every fetch tool here publishes an `sqt://` artifact and returns its id.

```
fetch_ohlcv_panel(tickers=[...], run_id="study7", name="bars")
        │
        ▼
sqt://price_panel/study7/bars
        ├──→ research      (indicators, factors, cointegration)
        ├──→ backtest      (signal panels, custom strategies)
        ├──→ portfolio     (construction, risk)
        └──→ modeling      (dataset construction)
```

A reference crosses runtimes and processes, survives two agents that cannot
see each other's context, and shows up in the decision log as an input to
whatever consumed it. Returning the frame inline would put megabytes into a
conversation that then carries them on every subsequent turn — a
2,000-ticker daily panel is not a thing to paste into a message.

**Pick the shape the consumer needs.** `fetch_returns_panel` gives a wide
date-by-ticker frame, which PCA, correlation, factor regression and
portfolio construction consume directly. `fetch_ohlcv_panel` gives stacked
long bars with an `entity` column, which is what indicator and backtest work
wants. Fetching the wrong one means the consumer rebuilds it, which is the
waste this runtime exists to remove.

## What the data cannot support is part of the answer

Four silences that used to be invisible, now returned in `warnings`:

**A missing ticker is absent, not NaN.** A universe fetch drops names that
returned nothing. They are named in `warnings`, because a complete-case join
downstream will not see they were ever requested — the panel simply looks
like a smaller universe that someone chose.

**A truncated tape is not a short one.** `fetch_tick_tape` and
`fetch_quote_panel` take a `limit`, and hitting it means the window is
incomplete. Every rate and total computed from a truncated tape understates
the real one, so the result says the cap was reached rather than leaving the
number to look like a measurement.

**Quotes are top of book.** No shipped provider exposes depth, so queue
position and resting size at a level are not in the data and cannot be
inferred from it.

**A provider that is not point-in-time hands back restated values under
their original dates.** `get_dataset_metadata` reports that, along with
whether prices are adjusted and whether the universe is survivorship-free.
A backtest joining on those dates is using information nobody had, and the
warning says so in those terms rather than as a flag.

## Temporal contracts, and the two ways to get one

| | What it reads | What it can say |
|---|---|---|
| `get_dataset_metadata` | The provider's own declaration | What the source GUARANTEES |
| `infer_temporal_contract` | The frame's columns | What is PRESENT in this frame |

Inference is the weaker of the two and is honest about being weaker. Reading
columns can only tell you what is there — a frame that happens to contain no
restatement is indistinguishable from one whose source discards them, and
inference correctly refuses to guess between those. Use it for data this
library did not fetch: a vendor extract, another system's output. Prefer the
provider's declaration whenever there is one.

## Bundles

A frame is half a fact. The other half is what its source can say about when
each row became knowable, and `join_point_in_time` later depends on exactly
that pairing.

```python
build_data_bundle(
    frames=[
        {"frame_kind": "bars",         "ref": "sqt://price_panel/...", "source": "yfinance"},
        {"frame_kind": "fundamentals", "ref": "sqt://...",             "source": "vendor"},
    ],
    run_id="study7", name="inputs",
)
# -> sqt://data_bundle/study7/inputs
```

**A bundle holds references, never copies.** It cannot diverge from the
frames it names, and assembling one is therefore free. That is also why
there is no `add_bundle_frame`: a mutable bundle would let two callers
disagree about which version was the one that got validated.

`validate_data_bundle` returns a verdict rather than raising, because the
answer is usually "yes, with caveats" and a caller needs the caveats to
decide.

**`require_pit` defaults to `false`, and that is not a lowered bar.** No
shipped provider reports `point_in_time=True` for every frame kind, so
requiring it would refuse essentially every bundle. A `usable` verdict at
the default therefore does **not** mean a leakage-free join is possible —
set `require_pit=true` when that is the actual question.

## Ratios, and the difference between a unit and a definition

`compare_ratio_frames` takes two providers' values as arguments — so it
works for sources this library cannot fetch — and **classifies** each
disagreement rather than only measuring it.

That distinction is the whole tool. A unit mismatch is fixable by rescaling.
A definition difference is not, and averaging across one produces a number
neither provider would stand behind. Reporting only a percentage gap leaves
the reader unable to tell which they are looking at.

`validate_financial_ratios` flags values implausible on their face. It is a
weak signal in one direction only: it catches ratios that are obviously
wrong, never ratios that are merely incorrect.

## What is deliberately not here

**Data quality checks.** `get_data_quality_report` in `research` already
reports missing bars, stale prices and price jumps. A second name for those
would be exactly the confusable duplication the runtime split exists to
prevent.

**Order book FETCHING.** `DataProvider.get_order_book` is still a declared
contract with canonical columns and no implementation — no shipped provider
serves depth, and a tool that always refuses is worse than no tool.

That was never the only way to have a book, though, and only fetching was
ever blocked. `register_external_dataset` takes depth you already hold — a
vendor extract, an ITCH replay — and makes it resolvable without copying it,
and `get_order_book_metrics` reads it through that reference. The analytics
were written and tested against the column contract long before a source
existed. What arrived is the source, not the analytics.

## Data this library did not fetch

Every other tool here fetches. These three do the opposite, and they exist
because the fetch path has a ceiling the rest of the surface never has to
notice.

A provider call returns one whole `pd.DataFrame`, and `publish` then writes
a second complete copy under `SQT_RUNS_DIR`. Two full materializations of
the same bytes is fine for a decade of daily bars and impossible for an
afternoon of L2 depth. The only concession to size anywhere else is
`fetch_tick_tape`'s `limit`, which does not sample — it **truncates**, so
every rate and total computed downstream understates the real one and
nothing in the numbers says so.

So `register_external_dataset` takes a Parquet or CSV file — or a directory
read as one partitioned dataset — and stores a POINTER and a schema. Nothing
is copied. Resolving the reference returns an `ExternalDataset` handle
rather than a frame, deliberately: something frame-shaped would let a
consumer written for a fetched panel pull forty gigabytes through an `.iloc`
without anyone deciding to. Rows come out through `batches()`, and column
projection means reading four columns of a sixty-column book reads four.

| kind | what it holds |
| --- | --- |
| `order_book_panel` | L2 depth snapshots, the shape `get_order_book_metrics` reads |
| `event_panel` | Rows carrying `event_time` and `available_time` |
| `tick_tape` | Trades, with `price` and `size` |
| `quote_panel` | Top of book, with `bid_price` and `ask_price` |

The first two are external-only — nothing in this library produces one, so
there is no in-memory publish path to preserve. The other two exist both
ways on purpose: a tape `fetch_tick_tape` fetched and a tape bought from a
vendor are the same content addressed differently, and one kind with two
storages beats an `external_tick_tape` that would double the taxonomy and
let a consumer accept one while refusing the other.

### What registration does not promise

That the file is still there, or still the same bytes, when someone resolves
the reference. A published artifact is immutable because this library wrote
it and refuses to overwrite it. An external file belongs to you and can be
re-extracted underneath a live reference.

So `describe_external_dataset` reports `changed_since_registration`, which
has no equivalent anywhere else on this surface. The check behind it is a
**fingerprint** — a digest of every file's name, size and mtime — spelled
with a different key from a published artifact's `content_hash` because it
is a weaker claim. It catches a re-extract, a truncated copy and a partially
written file. It does not catch an edit preserving both size and mtime.
Hashing the bytes would catch that too and would cost the full read this
whole path exists to avoid, so the weaker check that always runs beats the
stronger one nobody would wait for.

### Schema at registration, rows at validation

`register_external_dataset` checks columns and refuses a book with no
`ask_size_0` before anything reads a row. It does **not** read the rows, and
that gap is deliberate rather than lazy: a book with its bid and ask columns
transposed has a perfectly valid schema. Every column is present, every
value is a real price, and only the ORDER is wrong.

`validate_external_dataset` is what catches that, scanning in batches and
returning a verdict rather than raising — because three crossed books in
nine million rows is fine and a third of them crossed is transposed columns,
and only a count separates the two. It is the out-of-core sibling of
`validate_pit_records`, which is capped at 5,000 rows passed inline through
a tool call's JSON, and it calls the same `validate_pit_frame` checks per
chunk rather than reimplementing them. The scan stops at `scan_limit` and
says so; every count is reported against what was scanned, never against a
total the scan never reached.

### Databento

A raw Databento export does not satisfy these contracts, and the refusal
names the normalizer to run rather than leaving you to work it out. It
spells `bid_price_0` as `bid_px_00`, and it carries two things that would
each produce a confident wrong number:

- **Fixed-point int64 prices**, scaled by 1e-9. A raw `bid_px_00` of
  `100_010_000_000` is $100.01. Registered unscaled, every spread and
  microprice is off by a factor of a billion — and stays finite and ordered,
  so nothing looks broken.
- **`UNDEF_PRICE` for an empty level**, which is int64 max. Scaled, that is
  a **$9.2 billion quote** — so sentinels are masked *before* scaling,
  after which a sentinel is just a large float and no longer identifiable.

`standard_quant_tools.data.databento` handles both. It also drops trailing
levels that are empty in every snapshot — left in, the dataset would declare
ten levels while four hold nothing and `depth_slope` would regress against
them — keeps `action`, `side`, `flags` and the per-level order counts a
naive rename discards, and reports which of `ts_recv` and `ts_event` it
used. Those two differ by the network, which is exactly the quantity a
latency study measures, so the choice is never left implicit.


## The tools

| Tool | Answers |
|---|---|
| `fetch_ohlcv` | One symbol's bars, as a `price_panel` reference |
| `fetch_ohlcv_panel` | A universe's bars, stacked long with an `entity` column |
| `fetch_returns_panel` | A wide date-by-ticker return frame, ready for panel analysis |
| `fetch_tick_tape` | Individual trades, for measuring rather than estimating |
| `fetch_quote_panel` | Top-of-book quotes, what Lee-Ready signing needs |
| `fetch_financial_ratios` | A company's ratios, with implausible values flagged |
| `get_dataset_metadata` | What the provider guarantees: adjusted, survivorship, point-in-time |
| `infer_temporal_contract` | What a frame's own columns imply about timing |
| `build_continuous_futures_series` | Stitch a futures chain into one series, returning the back-adjusted research series and the tradeable contract map SEPARATELY -- an adjusted price is fine for indicators and is not a price anyone could have traded |
| `build_data_bundle` | Name several published frames as one unit |
| `describe_data_bundle` | What a bundle contains and what its sources promise |
| `validate_data_bundle` | Is this safe to model on, and what blocks it |
| `validate_financial_ratios` | Check ratios you already hold, without fetching |
| `compare_ratio_frames` | Two sources side by side, each gap classified |
| `register_external_dataset` | Make a file you already hold resolvable, without copying it |
| `describe_external_dataset` | Its schema and size, and whether it changed since registration |
| `validate_external_dataset` | Scan it in batches for what would produce wrong numbers |

Full argument lists: [20_tool_index.md](20_tool_index.md#data--data).

## Related

- [01_data_fetching.md](01_data_fetching.md) — the providers themselves
- [11_data_quality.md](11_data_quality.md) — the quality checks, which live in `research`
- [15_modeling.md](15_modeling.md#the-temporal-contract) — where bundles are consumed
- [19_runtimes.md](19_runtimes.md) — why this is its own execution boundary
