# The Data Runtime

Fourteen tools for getting the bytes and saying what they can support.

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

**Order books.** `DataProvider.get_order_book` is a declared contract with
canonical columns and no implementation — no shipped provider serves depth.
A tool that always refuses is worse than no tool, so `get_microprice`,
`get_depth_profile` and the rest wait on a real L2 provider rather than
shipping as guaranteed failures.

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
| `build_data_bundle` | Name several published frames as one unit |
| `describe_data_bundle` | What a bundle contains and what its sources promise |
| `validate_data_bundle` | Is this safe to model on, and what blocks it |
| `validate_financial_ratios` | Check ratios you already hold, without fetching |
| `compare_ratio_frames` | Two sources side by side, each gap classified |

Full argument lists: [20_tool_index.md](20_tool_index.md#data--data).

## Related

- [01_data_fetching.md](01_data_fetching.md) — the providers themselves
- [11_data_quality.md](11_data_quality.md) — the quality checks, which live in `research`
- [15_modeling.md](15_modeling.md#the-temporal-contract) — where bundles are consumed
- [19_runtimes.md](19_runtimes.md) — why this is its own execution boundary
