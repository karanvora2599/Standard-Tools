# Databento against the live market: what two suites found

A record of the first live tests this library has had against Databento,
the three defects they found, and the checks that passed.

**Status: the suites are merged as `941a730` (2026-09-20), 47 tests in
`tests/data/test_databento_live.py` (29) and
`tests/data/test_databento_pipeline_live.py` (18). 46 pass, one is a
strict xfail recording F1.** Every claim below was measured against the
live feed on 2026-09-20 with the source read at `d04e84e`. No fix is
applied: F1 and F2 change numbers this library has already produced, and
that is a decision to take deliberately rather than inside a test commit.
The baseline before the suites was 7,228 offline tests passing with two
pre-existing failures; after, 7,267 passing with the same two, and 47 more
deselected from the default selection.

The short version: the provider parses correctly and the prices are right.
The **volume** is not, on the default path, by a factor of about thirty —
and nothing in the returned frame says so.

---

## 1. Why there were no live tests, and why that was half right

`tests/data/test_databento_provider.py` opens by saying every test in it is
offline, deliberately, because dataset preference, the finalization
walk-back and entitlement memory "are exactly the parts that are expensive
to get wrong and impossible to exercise against a live API in a suite."

That is correct about those parts. They are logic, logic is tested with an
injected client, and a live API would make them flaky rather than better
covered. Nothing below argues for changing that file.

What an injected client cannot test is whether the bytes coming back are
the market. A provider that parses a fixture perfectly and returns three
percent of the volume passes every offline test ever written, because the
fixture was built from the same assumption as the parser. That is exactly
what was happening, and F1 is what it looks like.

So the live suites check only the things a fixture cannot, in three ways
that never take the vendor's word for itself:

1. **Invariants the market guarantees.** A low is not above an open. A
   book is not crossed. A trade prints inside its own session. No second
   source needed.
2. **A second vendor.** Closes and volumes joined against yfinance —
   different infrastructure, different business, no shared upstream.
3. **Cross-schema agreement.** The depth feed and the daily bars are
   different products from different pipelines, so every quote in the book
   must sit inside that session's own high and low. This is the one that
   catches a price-scaling error, which is the failure `data/databento.py`
   exists to prevent and the one a fixture can never demonstrate.

---

## 2. Defects found against the live feed

### F1. `DATASET_CONSOLIDATED` is not the consolidated tape — *reproduced*

`data/databento.py:84` sets `DATASET_CONSOLIDATED = "EQUS.MINI"`, and
`:88-90` states the belief plainly:

> EQUS.MINI is the consolidated tape and is the best answer whenever it
> covers the window

`databento_provider.py:250` puts it first in the bar preference, so it
serves nearly every bar this library produces.

It is not the consolidated tape. Measured over 2026-08-03..2026-09-18
against yfinance's consolidated daily bars, on four symbols:

| dataset | volume ÷ consolidated | median close error |
|---|---|---|
| EQUS.SUMMARY | **1.0000** on all four | **0.000000** |
| XNAS.BASIC | 0.5789 – 0.9392 | 0.001319 |
| XNAS.ITCH | 0.1360 – 0.3099 | 0.001403 |
| **EQUS.MINI** | **0.0227 – 0.0355** | 0.000935 |

Per symbol, EQUS.MINI: AAPL 0.0330, MSFT 0.0292, SPY 0.0355, TSLA 0.0227.
33 joined sessions per symbol. The same sweep over 2026-09-08..2026-09-19
gives EQUS.MINI 0.0217 – 0.0347 and EQUS.SUMMARY 1.0000 again, so the
result is the dataset's nature rather than one window's.

The prices are fine. That is what makes this dangerous rather than
obvious: the frame looks entirely healthy — right shape, right index,
right closes to a tenth of a percent — and every volume-weighted number
computed from it is wrong by more than thirty times. VWAP, average daily
volume, dollar-volume ranks, liquidity screens, volume breakouts,
participation-rate sizing and any turnover constraint in the backtest all
inherit it silently. A liquidity filter written as "trades more than a
million shares a day" keeps a different universe than its author believes.

EQUS.SUMMARY is the dataset that matches consolidated volume, share for
share, on every symbol tried, and its closes match to zero error.

**The Carbon engine has the same defect from the same cause.** It pins
`DATABENTO_OHLCV_DATASET=EQUS.MINI` in `services/engine/.env`. Measured
live through its own bars provider: volume **0.0315** of consolidated,
close error 0.0011. Its chart volume bars are wrong wherever Databento
serves them.

**Fix.** EQUS.SUMMARY for `ohlcv-1d`. It is not a rename: EQUS.SUMMARY
carries only `ohlcv-1d`, `definition` and `statistics`, so `_bar_datasets`
has to become schema-aware — daily prefers EQUS.SUMMARY, intraday cannot
use it at all and needs its own preference, where the honest answer is a
venue feed with its share stated rather than a sample feed presented as a
tape. Because the change moves every volume number this library has
produced, it is recorded as a strict xfail
(`test_the_default_bars_carry_consolidated_volume`) naming the fix, rather
than made quietly. Remove the xfail with the change.

### F2. Which feed answered depends on the window, and is never reported — *reproduced*

`_bar_datasets` returns a preference list and `_range`
(`databento_provider.py:282-300`) declines a dataset whose coverage does
not contain the whole requested window, so the request falls through.
`CONSOLIDATED_START` is 2023-03-28.

One call is therefore served by exactly one dataset, which is the right
design and means there is no discontinuity *inside* a frame. Confirmed: a
call spanning 2023-02-01..2023-05-31 is served whole by XNAS.ITCH, with no
step at the boundary.

The problem is between calls. Two adjacent windows are served by different
feeds:

```
2023-02-01 .. 2023-03-24   served by XNAS.ITCH    (~30% of consolidated)
2023-03-29 .. 2023-05-31   served by EQUS.MINI    (~3% of consolidated)
```

A caller who fetches in chunks — which is what a long backtest, a cache
fill, or any per-year loop does — splices two feeds whose volumes differ
by an order of magnitude, at a date fixed by the vendor's coverage rather
than by anything in the market. The result is a synthetic structural break
that no corporate action explains and that a regime detector, a volume
z-score or a turnover model will happily fit.

And the caller cannot detect it. Every accessor discards the dataset:

```
databento_provider.py:424   frame, _dataset = self._fetch(...)   # get_ohlcv
databento_provider.py:484   frame, _dataset = self._fetch(...)   # get_trades
databento_provider.py:505   frame, _dataset = self._fetch(...)   # get_quotes
databento_provider.py:545   frame, _dataset = self._fetch(...)   # get_order_book
databento_provider.py:583   frame, _dataset = self._fetch(...)   # get_order_events
```

`get_metadata` (`:619-643`) cannot help either: it is a static self-report
built from constants, with no dataset field and no knowledge of what any
particular call reached.

**Fix.** Report the dataset that served. `frame.attrs["dataset"]` is the
cheapest version and survives most pandas operations; a field on
`DataSetMetadata` is the honest one, but it has to be per-call rather than
per-provider to mean anything. Either way, add a warning when a request
falls through to a feed whose volume basis differs from the preferred
one — a caller who asked for ten years and got two feeds should be told
once, not never.

### F3. A multi-publisher dataset returns one row per publisher, undeduplicated — *reproduced*

`_to_ohlcv` (`databento_provider.py:428-461`) validates columns, decides
the price scale, coerces the index and returns `out.sort_index()`. It
never collapses duplicate timestamps.

DBEQ.BASIC publishes a bar per publisher per session. Through the
provider, `get_ohlcv("AAPL", "2026-09-14", "2026-09-18")` on that dataset
returns **15 rows for 5 sessions**, `index.is_unique` **False**, three
rows per day with different closes (332.54 / 332.41 / 332.79 on
2026-09-16) and volumes an order of magnitude apart (27,329 / 903,206 /
183,252).

A caller computing returns gets **14 returns from a 5-session window**,
most of them cross-publisher noise rather than market moves. Anything
using `.loc[date]` gets a Series where it expected a scalar.

Severity is bounded by reachability: DBEQ.BASIC is not in the default
preference (`['EQUS.MINI', 'XNAS.BASIC', 'XNAS.ITCH']`), and the four
datasets that are all return one row per session. It is reachable through
`DATABENTO_OHLCV_DATASET=DBEQ.BASIC`, or through the `dataset=` /
`depth_dataset=` constructor arguments. So this is a configuration hazard
rather than a live defect — but it is reachable by configuration alone,
with no error and no warning.

**Fix.** `_to_ohlcv` should refuse or aggregate, not silently pass through.
Refusing is more in keeping with the rest of this provider: a frame with a
non-unique index is not an OHLCV series, and the caller needs to know
which publisher they meant. If aggregating, it is last-close and
summed-volume per session, and the payload has to say it aggregated.

---

## 3. Checked, and not a defect

**CAGR's denominator.** The obvious reference — `len(series) / 252` —
disagreed with `return_metrics.cagr` in the fourth decimal (0.226549 vs
0.227039 on two years of AAPL). The library is right and the reference was
wrong: N closes span N−1 return intervals, and `return_metrics.py` had
already reasoned that out in its own comment, noting the error is
negligible on a decade and 5% on a one-month window. The test now asserts
the correct convention **and** asserts the two conventions differ on this
window, so it cannot pass by coincidence.

**Price scaling.** The headline claim of `data/databento.py` — that scale
is decided from the dtype and cross-checked in both directions, with
sentinels masked before scaling — holds on real data. No quote in a real
MBP-10 book exceeded 1e6 (an unmasked `UNDEF_PRICE` would be ~9.2e18), no
daily close moved more than 35% session to session across two years, and
every book quote sat inside its own session's high and low.

**The book itself.** Ten seconds of AAPL at the open: 800 snapshots, **0**
crossed, bids strictly descending and asks strictly ascending on
**800/800** rows.

---

## 4. Pre-existing failures, unrelated to data

`tests/surface/test_adversarial_inputs.py::TestTheBaselineHolds` fails two
tests, before and after this work, unchanged:

- `test_every_tool_without_a_baseline_is_declared` — four modeling tools
  (`validate_model_spec`, `build_model_dataset`, `run_model_experiment`,
  `run_feature_ablation`) reject their own synthesized input on a
  cross-field validator, so they sit in no adversarial or determinism
  check. Two distinct causes: an unknown preprocessing step `'a1'`, and
  `missing.policy='drop'` carrying fields that belong to
  `forward_fill_bounded`.
- `test_the_synthesizer_covers_most_of_the_surface` — 205 of 209 tools
  synthesized, 0 declared unsynthesizable.

Collection counts drift between runs because that file synthesizes tests
from the live tool surface; the offline total moved 7,317 → 7,376 between
two runs with no source change to the tests themselves.

---

## 5. Traps that make a live test lie

Recorded because each one produces a **passing** test that checks nothing.

**Inherited dataset pins.** `DatabentoProvider` reads
`DATABENTO_DATASET`, `DATABENTO_DEPTH_DATASET` and
`DATABENTO_OHLCV_DATASET` from the environment
(`databento_provider.py:154-164`, `:246`). The Carbon engine's `.env` sets
all three. A first pass of this work concluded the `dataset=` constructor
argument was ignored — it was not; the inherited `DATABENTO_OHLCV_DATASET`
override was winning, and the conclusion was an artifact of whose `.env`
had been loaded. Both suites now clear all three in a module-scoped
autouse fixture and name their dataset outright.

**yfinance `end` is exclusive; Databento's is inclusive.** The same two
dates name windows differing by one session, and the last one reads as a
session Databento invented. This produced the only genuine failure of the
first run.

**yfinance is tz-naive; Databento is UTC-aware.** Joining without
normalising raises, or worse, silently yields an empty frame — and a test
written on an empty join passes by comparing nothing to nothing. Both
suites assert a minimum joined row count before comparing anything.

**Bollinger columns are prefixed** (`BB_Upper`, not `upper`). Asserting
the label rather than stripping it makes the test about naming.

---

## 6. What the live feed confirmed

The depth measures in `analysis/order_book.py` were written and tested
against synthetic books because nothing could serve a real one. They now
run on ten seconds of a real Nasdaq open and produce a market:

| measure | value | reads as |
|---|---|---|
| mean touch spread | 1.60 bps | a penny on a $333 stock is ~0.3 bps |
| book updates | 130 /sec | a real ITCH feed |
| mid changes | 14 /sec | |
| order-flow imbalance | 1,265 over 6.16 s | |
| depth profile | size rises 67 → 110 from level 0 to 4 | books rest more size away from the touch |
| crossed snapshots | 0 of 800 | |

The microprice stayed inside the touch on every snapshot where it was
defined, and the depth profile's per-level distances came out monotonically
increasing — which a synthetic book has to be told to do and a real one
does on its own.

Cross-vendor, over 34 sessions: median close disagreement **0.13%**, worst
**0.88%** — consistent with a last-print difference between tapes rather
than a different number. Annualised volatility and worst drawdown computed
from each vendor's series agreed within the suite's 3% and 5% bands.

---

## 7. Cost

Priced through `metadata.get_cost` and `get_billable_size` before any test
was written. A full live pass moves **20.06 MB** billable, dominated by
the single 10-second MBP-10 book (18.1 MB); every other fetch is under
2 MB and the daily-bar fetches are ~2 KB each. `get_cost` returns **$0.00**
and `list_unit_prices` returns empty for these datasets on this account,
which is a subscription rather than metered per gigabyte.

Fixtures are module scoped so a pass fetches each window once. Both files
are `pytest.mark.integration` and skip without `DATABENTO_API_KEY`, so the
default selection (`-m "not integration"`) is untouched.

Run them with the **engine's** interpreter — this repo's `.venv311` does
not have `standard_quant_tools` installed:

```
cd "C:/Users/karan/Documents/Projects/Standard Tools"
DATABENTO_API_KEY=... \
"C:/Users/karan/Documents/Projects/Carbon Redifined/services/engine/.venv/Scripts/python.exe" \
  -m pytest tests/data/test_databento_live.py tests/data/test_databento_pipeline_live.py \
  -q -m integration -p no:randomly
```

---

## 8. Not covered yet

Named rather than implied, so the gaps are known:

- **`get_order_events` (MBO).** Fetched during exploration and shaped
  correctly, but no assertions were written for order-lifecycle
  consistency (an add before its cancel, a fill not exceeding its
  resting size).
- **Futures and options.** GLBX.MDP3 and OPRA.PILLAR are entitled on this
  account and untested here; the continuous-contract stitching in
  `data/continuous.py` is exactly the kind of logic real data would
  stress.
- **The entitlement-denial path.** Tested offline as logic; not provoked
  live, because doing so needs a dataset the key genuinely lacks.
- **The backtest engine over real bars.** The suites stop at indicators
  and metrics. A walk-forward over two years of real prices, with costs
  and turnover, is the next thing worth adding — and it is the first
  consumer that F1's volume error would visibly distort, through
  participation limits and liquidity constraints.
- **Intraday bars.** Only `ohlcv-1d` is exercised. The `1s`/`1m`/`1h`
  schemas share the dataset preference and therefore F1 and F2.
