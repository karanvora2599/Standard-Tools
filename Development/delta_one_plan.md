# Delta One — the tenth runtime

> **Status: PLAN. Nothing below has shipped.**
>
> Written against `main` at 178 tools across nine runtimes, after a
> six-agent survey of what already exists. Every file path, line number and
> measurement here was verified against the working tree rather than
> recalled — where a claim could be run, it was run, and the result is
> quoted.

The thesis is not that Delta One needs building from nothing. It is that
**the mathematics is largely present and the economics is entirely
absent.** The library can price a forward, regress a beta, estimate an
effective spread and itemize a trade's cost. It cannot say that a future is
46 bps rich, that the hedge is 903 contracts, or that a TRS is the cheaper
of five ways to hold the same exposure. Those are relationships *between*
instruments, and nothing in the surface models an instrument at all.

---

## 1. What the survey actually found

### 1.1 The carry equation exists and is good

`analysis/derivatives.py:943` — `implied_forward_price()`:

```python
def implied_forward_price(
    *, spot, time_to_expiry, risk_free_rate,
    dividend_yield=0.0, borrow_rate=0.0,
) -> Dict[str, Any]
# F = S * exp((r - q - b) * T)
```

It returns `spot`, `forward`, `net_carry_rate`, `basis`, `basis_pct`, and a
`components` dict splitting financing / dividend / borrow. **Those key names
are the repo's existing vocabulary for this — `delta_one` should match
them, not invent parallel ones.**

It stops one step short of being tradeable: it computes the theoretical
forward and never takes a *quoted* future as an argument. Its own warning
says so — "a quoted future differing from it is the market disagreeing about
one of the three components." Making that comparison is the whole of Tool 1.

### 1.2 Futures are genuinely greenfield

Zero hits repo-wide for `contract_size`, `tick_value`, `ContractSpec`,
`initial_margin`, `variation_margin`, `front_month`, `roll_yield`,
`continuous contract`, `open_interest`. Every `multiplier` hit is an ATR
stop scalar or a Bollinger width. `expiry` is always `time_to_expiry:
float`, in years, supplied by the caller — there is no expiry *date*
anywhere, and no contract calendar.

`contango` and `backwardation` **do** appear — but they describe an
implied-volatility term structure (`analyze_vol_term_structure`), not a
price curve. Reuse the vocabulary; be explicit that `futures.py` means the
price curve, or the two will be confused.

### 1.3 Day count does not exist

Repo-wide grep for `year_fraction`, `day_count`, `ACT/360`, `30/360`,
`np.busday`, `BDay`: **zero hits.** What exists instead is five inline
`/ 365.0` sites (all ACT/365F, none reusable) and `TRADING_DAYS = 252`
duplicated as an independent literal in **seven** modules.

This is a hard prerequisite. Every carry number in this runtime is a rate
times a year fraction, and there is currently no year fraction.

### 1.4 The hedge mathematics is there; the translation is not

| Have | Path |
|---|---|
| `calculate_beta` → `{alpha, beta, r_squared}` | `analysis/regression.py:24` |
| `rolling_beta(window=60)` | `analysis/regression.py` |
| `multi_factor_regression` | `analysis/multi_factor.py` |
| `estimate_covariance(method="ledoit_wolf")` | `portfolio/covariance.py:53` |
| `marginal_risk_contribution` | `portfolio/construction.py` |
| `structural_break_test(series, break_index, regressor=)` | `analysis/diagnostics.py` — tests whether a **hedge ratio** broke |

Missing: **`tracking_error` is not a function.** It is a local variable
inside `information_ratio` (`metrics/risk_metrics.py:188`). Nothing else in
the repo computes it. And there is no beta-hedge-ratio-against-a-benchmark;
`min_volatility` is adjacent but is absolute-risk, not benchmark-relative.

### 1.5 The cost model composes, but its vocabulary is closed

`backtest/costs.py` is ten pure functions returning currency (except
`sqrt_impact_bps`, the one bps outlier). They compose by summation and
nothing else — there is no aggregator object.

The blocker is one line. `agent/models.py:3699`:

```python
component: Literal["commission", "spread", "impact", "borrow", "margin_interest"]
```

A futures financing leg, a roll leg, or a swap funding leg cannot be
expressed. **Decision: `delta_one` defines its own leg model rather than
widening this one** — see §4.4.

### 1.6 The portfolio engine cannot hold a future

Two variables are the entire book (`backtest/portfolio_engine.py:714`):

```python
cash = initial_capital
shares_vec = np.zeros(n_tickers, dtype=np.float64)
```

and the identity `position value == shares × price == cash paid` is baked
into eight sites, of which two are fatal:

- `cash -= delta * price` (line 867) — a future moves no cash beyond margin
- `equity = cash + shares_vec @ close_prices` (814, 944, 1130) — a future
  has no market value once variation margin is credited to cash

This is why futures backtesting is **Phase III and lives in `backtest`, not
`delta_one`**. The economics and the simulation are different jobs.

### 1.7 Data: Bloomberg passes futures identifiers through, and that is all

Verified by running `_to_bloomberg_ticker`:

```
'ESZ5 Index'    -> 'ESZ5 Index'            (pass-through works)
'SPX Index'     -> 'SPX Index'
'CL1 Comdty'    -> 'CL1 Comdty'
'ESZ5'          -> 'ESZ5 US Equity'        <-- bare code becomes a US equity
'SPX INDEX'     -> 'SPX INDEX US Equity'   <-- match is CASE-SENSITIVE
```

Three consequences: normalize case before handing symbols in; the
non-suffix part is never validated; and timezone metadata reports
`America/New_York` for a CME contract because `_bloomberg_timezone` needs
three tokens to read a yellow key.

Also confirmed absent: index constituents, dividend history, corporate
actions, open interest, and intraday bars (`IntradayBarRequest` is not
implemented; only `1d`/`1wk`/`1mo`).

**Therefore Phase I takes structured arguments, never fetches.** This is the
same call the derivatives runtime made about option chains, and it is
test-enforced: `test_no_description_promises_a_data_source_the_library_lacks`.

### 1.8 Four live bugs, found and fixed in this pass

Each was verified by running it, then fixed. All four share a shape worth
naming: **nothing failed.** Three degraded silently and one was a test that
looked exhaustive.

1. **`implied_forward_price` did not bound `dividend_yield`.** It was the
   one term in `r - q - b` passed straight to `math.exp` unchecked.
   `q=1e5` returned `forward=0.0` silently; `q=-800` escaped as a bare
   `OverflowError` rather than a `ValidationError` naming the argument. Now
   routed through `_option_inputs` like the other two rates. This one
   matters here because `carry.py` leans on this function.

2. **`agent/runtimes/_shared.py` clobbered its own C++ import.** The
   `_cpp_core = None; HAS_CPP = False` initializers sat *after* the
   try/except and ran unconditionally, so `HAS_CPP` was `False` on every
   machine, built extension or not. The fused technical-indicator fast path
   in `research/tools.py:283` that reads these names was dead code
   wherever it was imported from here. Now `True` where the extension
   exists.

3. **`DEEP_DOCS` in `generate_tool_index.py` silently omitted `data`.** The
   lookup degrades to an em-dash, so a runtime with a guide on disk
   advertises none and nothing fails. `26_data.md` had been invisible in
   the generated index; `meta` also pointed only at the audit guide while
   `27_meta.md` was its actual deep dive. Both fixed and the index
   regenerated. **This is the single easiest item in §5 to forget, and the
   only one that stays green when you do.**

4. **`test_scoping_does_not_change_which_tools_a_runtime_has` checked one
   runtime, not nine.** Its `assert` was dedented out of the
   `for runtime in ALL_RUNTIMES` loop, so eight servers were built and
   discarded and only whichever runtime sorted last was verified. Re-indented.
   It passes for all nine — the loop was right, it just was not running.

---

## 2. Scope

> Pricing, relative value, hedging, replication, carry, financing and
> execution of instruments whose value moves approximately one-for-one with
> an underlying equity or index.

The boundary against `derivatives` is clean and worth stating in the
runtime description, because an agent will otherwise route badly:

| | `derivatives` | `delta_one` |
|---|---|---|
| Question | What is this contract worth, and what does holding it do to me? | Which instrument is the cheapest way to own or hedge this exposure? |
| Object | One option, convex | Many instruments, linear |
| Risk | Greeks, vol surface | Basis, carry, tracking error |

They meet at exactly one place, and it is a feature: options give a
synthetic forward via put-call parity, and that forward is one row in
`compare_delta_one_expressions`.

---

## 3. Phase I — nine tools

**Nine, not eight.** `MINIMUM_RUNTIME_SIZE = 8` is pinned in two files
(`tests/surface/test_invariants.py:30`, `tests/agent/test_runtimes.py:67`).
Shipping exactly eight leaves zero margin: if one tool proves not to belong
during review, the runtime becomes unshippable and the work stalls. The
ninth — `analyze_basis_history` — is the cheapest tool on the list, being
almost pure composition over `spread_zscore`, `half_life` and
`run_stationarity_tests`, all of which already exist.

| # | Tool | Library module | Reuses |
|---|---|---|---|
| 1 | `analyze_cash_futures_basis` | `basis.py` | `implied_forward_price` |
| 2 | `solve_forward_carry` | `carry.py` | inverse of the same identity |
| 3 | `analyze_basis_history` | `basis.py` | `spread_zscore`, `half_life`, `run_stationarity_tests` |
| 4 | `analyze_futures_curve` | `futures.py` | new; mirrors `analyze_vol_term_structure`'s shape |
| 5 | `analyze_roll` | `futures.py` | `costs.py` |
| 6 | `size_futures_hedge` | `hedging.py` | `calculate_beta` |
| 7 | `analyze_hedge_effectiveness` | `hedging.py` | `rolling_beta`, `structural_break_test`, new `tracking_error` |
| 8 | `analyze_index_basket` | `baskets.py` | new N-leg; `pairs.py:208` solves only 2 legs |
| 9 | `compare_delta_one_expressions` | `expressions.py` | all of the above + `costs.py` |

### 3.1 Naming

Checked all fifteen proposed names against the live registry with
`difflib`. No collisions, and the highest cross-runtime similarity is
**0.76**, comfortably under the 0.95 that
`test_the_two_names_are_no_longer_confusable` fails on.

But one name is a semantic collision the ratio does not catch:

> `calculate_delta_one_hedge` (0.76) vs the existing `simulate_delta_hedge`

Both read as "delta hedge". One sizes a futures hedge for a cash portfolio;
the other simulates rehedging a short option. An LLM choosing between them
on name alone will get it wrong, and the repo has scar tissue here — the
`analyze_feature`/`analyze_features` pair at 0.97 forced a rename.

**Renamed to `size_futures_hedge`.** It says what it does, names the
instrument, and drops "delta hedge" entirely.

### 3.2 Tool 1 — `analyze_cash_futures_basis`

The one that makes the existing forward calculator tradeable.

```
observed carry  c_mkt  = ln(F_mkt / S) / T
fair carry      c_fair = r - q - b
basis spread           = c_mkt - c_fair
```

Returns both the point basis and the annualized bps spread, the
classification (`future_rich` / `future_cheap` / `fair`), and the
financing / dividend / borrow decomposition carried through from
`implied_forward_price` so the caller can see *which* component the market
disagrees on.

### 3.3 Tool 2 — `solve_forward_carry`

One inverse solver, not three tools. `solve_for: Literal["financing_rate",
"dividend_yield", "borrow_rate"]`, rearranging `ln(F/S)/T = r - q - b`.

Three tools would triple the surface cost for one identity and force the
agent to pick among near-identical names — exactly the failure §3.1 is
guarding against.

### 3.4 Tool 6 — `size_futures_hedge`

The tool that turns every regression in the library into a position:

```
dollar beta      = V_p × β_p
contract exposure = F × multiplier × β_f
N                = -(V_p × β_p) / (F × multiplier × β_f)
```

Must return **both** `contracts_exact` and `contracts_rounded`, and the
`residual_dollar_beta` that rounding leaves behind. Reporting only the
rounded figure hides the residual, which is the number that decides whether
a second instrument is needed.

### 3.5 Tool 9 — `compare_delta_one_expressions`

The most agent-native tool on the list, and the reason the runtime is worth
having. Normalizes cash / ETF / future / synthetic forward / TRS onto one
annualized carry, itemized by financing, dividend, borrow, roll, spread,
commission, impact and capital requirement.

It moves the library from "here are some analytics" to "here are five
economically equivalent ways to hold this, and here is why they differ."
The LLM should not be reconstructing the financing convention, the
multiplier and the borrow basis in its head and comparing them — that is a
deterministic calculation and it should ask for the answer.

---

## 4. Architecture

### 4.1 Library package, not an `analysis/` module

```
src/standard_quant_tools/delta_one/
    __init__.py       # minimal; analysis/ does not re-export its newer modules either
    daycount.py       # year_fraction — the prerequisite (§1.3)
    contracts.py      # ContractSpec: metadata, NOT a tool
    carry.py          # forward, inverse solve
    basis.py          # spot-vs-quoted, basis history
    futures.py        # curve, roll
    hedging.py        # beta -> contracts, effectiveness, tracking_error
    baskets.py        # N-leg basket vs index
    expressions.py    # the comparison engine
```

A package rather than `analysis/delta_one.py` because it is eight modules of
new mathematics; `analysis/` modules are single files. Precedent:
`portfolio/`, `backtest/`, `modeling/` are all top-level packages.

Library functions are keyword-only, return `Dict[str, Any]` with a
`"warnings"` key, raise `ValidationError`, and use no scipy — matching
`analysis/`.

### 4.2 Runtime package mirrors `derivatives` exactly

```
src/standard_quant_tools/agent/runtimes/delta_one/
    __init__.py   # TOOL_DEFS -> TOOL_DISPATCH -> TOOL_CATEGORY, one source
    models.py     # ConfigDict(extra="forbid") on every input
    results.py    # Stat + _Result, extra="allow"
    tools.py      # thin wrappers, Result(**lib.fn(...)), no docstrings
```

Conventions that are load-bearing rather than stylistic:

- **`Stat` and `_finite_or_none` are copied, not imported.** Five modules
  carry their own. A shared one becomes the place cross-runtime coupling
  accumulates.
- **Results use `extra="allow"`; inputs use `extra="forbid"`.** The rule is
  splat ⇒ allow, field-by-field ⇒ forbid. Wrappers splat.
- **Enums are `Literal[...]`, never bare `str` or `Enum`.** Not style:
  `tests/surface/synth.py` gives a bare `str` the literal `"a"`, validation
  fails, and **the tool is silently dropped from the fuzz sweep** with no
  report.
- **Every float carries a bound.** `synth.py` midpoints those bounds to
  fabricate fuzz inputs. An unbounded float is an unfuzzed tool.
- **Result classes import at module top level, never under
  `TYPE_CHECKING`.** `mcp/catalog.py:202` calls `get_type_hints(fn)`; an
  unresolvable string annotation means no output schema.

### 4.3 TOOL_DEFS descriptions are the product

Measured across the twelve derivatives descriptions: **44–98 words, mean
75**, 2–5 sentences, exactly one paragraph, no newline, ≥60 chars.

The constraint that actually bites: `thin_description()`
(`mcp/catalog.py:324`) extracts **only the first sentence**, capped at 180
chars, and under `auto` thinning that is the entire description an agent
sees. Sentence one must therefore be self-contained and name the
deliverable — not a preamble.

Sentences two onward pre-empt the misreading and **state the failure mode
with a number attached**. Every one of the twelve does. That is the house
voice and it is what makes these tools choosable.

### 4.4 Delta One owns its own cost legs

`TradeCostLeg.component` is a closed five-value `Literal` in the portfolio
runtime. Two options: widen it, or define a new one.

**Define a new one.** Widening couples two runtimes through a shared enum
that each would then constrain, and the repo's consistent answer to that
trade is duplication — `Stat` is copied five times for exactly this reason.
`delta_one/results.py` gets `DeltaOneCostLeg` with its own wider `Literal`
covering financing, roll and swap funding. `backtest/costs.py` functions are
still called directly; only the leg *vocabulary* is local.

### 4.5 `get_implied_forward` does not move

It is the most Delta One tool on the existing surface, and it stays in
`derivatives` anyway.

Moving it would be a breaking change for anything scoped to `derivatives`,
and it earns its place there — put-call parity work needs it. `delta_one`
calls the same **library** function, `implied_forward_price()`. Sharing a
library function is not a boundary violation: the architecture scopes
*dispatch*, and says explicitly that values cross freely. Tool 1 supersedes
it for Delta One purposes regardless, since it does everything
`get_implied_forward` does plus the market comparison.

**Consequence: no `MOVED_FROM` entry, no breaking change, no donor
depletion.** `derivatives` stays at 12.

---

## 5. Wiring — seventeen sites

Verified against the tree. Miss one and the tool half-exists.

**Source (mandatory):**

| # | File | What |
|---|---|---|
| 1 | `agent/runtimes/__init__.py:86` | `RUNTIME_CATEGORIES["delta_one"] = ("delta_one",)` — **this one key enrols the runtime in ~35 sweeping tests** |
| 2 | `agent/runtimes/__init__.py:132` | `RUNTIME_LABELS` — `KeyError` in `_build()` without it |
| 3 | `agent/runtimes/__init__.py:144` | `RUNTIME_DESCRIPTIONS` — one newline-free paragraph |
| 4 | `agent/tools.py:~35, ~90, ~211` | import alias, name-by-name re-export, and `_RUNTIME_MODULES`. **Do not touch `__all__` here** — despite its comment it is a legacy partial list (`get_option_greeks`, `estimate_vpin` and others are already absent), and the derivatives commit did not extend it |
| 5 | `agent/__init__.py` | every tool name in the import block **and** `__all__`. This one *is* pinned, twice |
| 6 | `agent/router.py:34` | `TOOL_CATEGORIES["delta_one"]` — **omitting this is an import-time `RuntimeError`**; `_assert_categories_cover_every_tool()` runs at module scope, and `mcp/catalog.py` imports the module, so it kills the MCP server too |

The router description must clear pairwise Jaccard word-overlap < 0.35
against every existing category description (`tests/agent/test_router.py:338`).

**Not needed:** `agent/models.py` (put models in the runtime package),
anything under `mcp/` (all derived from `RUNTIME_CATEGORIES`), `handoff.py`,
`_shared.py`, `pyproject.toml`.

**Multi-agent and hardcoded test lists:**

| # | File | What |
|---|---|---|
| 7 | `Multi_Agent_Implementation/worker_agents.py:135` | one worker keyed exactly `"delta_one"` |
| 8 | `tests/agent/test_multi_agent_tool_coverage.py:148` | add to the literal worker set |
| 9 | `tests/agent/test_agent_tools.py:220` | add to hardcoded `known_categories` |
| 10 | `tests/agent/test_router.py:125` | ≥1 `EVAL_CASES` entry, or the routing eval silently stops covering the category |
| 11 | `tests/surface/test_invariants.py:243` | add a `(runtime, foreign)` scoping pair |
| 11b | `tests/agent/test_cross_runtime.py:68` | add `"delta_one"` to the provider-stub tuple **if it fetches**. `monkeypatch(..., raising=False)` makes omission silent at patch time and surfaces it later as a live network call in an unrelated test |

**Docs (test-enforced):**

| # | File | What |
|---|---|---|
| 12 | `Documentation/20_tool_index.md` | regenerate: `python Development/generate_tool_index.py` — byte-compared |
| 13 | `Documentation/19_runtimes.md:39,43-51` | "the nine runtimes" → ten; new table row `\| \`delta_one\` \| N \|` |
| 14 | `README.md:12` | `**178 LLM-callable tools**` → 187; `**nine parallel runtimes**` → `**ten**` |
| 15 | `README.md` cost table + guide table | new rows |
| 16 | `Documentation/18_mcp.md:148,435,539` | three hardcoded `178`s |
| 17 | `Documentation/28_delta_one.md` + `DEEP_DOCS` in the generator | the per-runtime deep dive, following `21_derivatives.md`. Backlink anchor format is `20_tool_index.md#<runtime>--<label>` |

Three further doc tables enumerate runtimes and **already omit `data`**, so
nothing is checking them: `18_mcp.md:379-388` (runtime → categories),
`18_mcp.md:278-287` (per-runtime detail costs), and
`13_agent_orchestration.md:116-128` (`TOOL_CATEGORY`). Only the
`19_runtimes.md` table has a test asserting row *presence*. Add the row in
all four, and fix `data` while there.

`CONTRIBUTING.md:118-121` carries the runtime-addition checklist and is
itself incomplete — it omits the numbered deep doc, the `DEEP_DOCS` entry
and the README runtime-table row. Amend it, or the next runtime repeats
this survey.

The precedent to read before starting is commit **`fec753f`**, "A
derivatives runtime" — the last brand-new runtime with a brand-new
category.

### 5.1 The verification that is not optional

```bash
python -c "from tests.surface.test_adversarial_inputs import TOOL_IDS; \
  print([i for i in TOOL_IDS if i.startswith('delta_one:')])"
```

`test_adversarial_inputs.py:91` wraps input synthesis in
`except Exception: continue`. A tool whose model `synth.py` cannot build is
**dropped from the fuzz sweep with no report**, and the only guards are
whole-surface floors (≥100 tools, ≥500 mutations) that a nine-tool gap will
not trip. The suite stays green with the entire runtime un-fuzzed.

`synth.py` matches on field-name heuristics — name fields `spot`, `price`,
`forward`, `prices`, `rate`, `yield` and synthesis works for free.

---

## 6. The MCP budget is no longer a constraint

The previous version of this analysis would have had to argue for headroom.
It no longer does: **the hard context ceiling was removed in this pass.**

`CONTEXT_CEILING_BYTES` was `180_000` and a test failed when the whole
surface crossed it. That number was chosen when the library had 54 tools,
and what it actually bounded was *tool count*, not cost — while
`--tool-detail auto`, which thins expensive schemas to fit `detail_budget`
and leaves every tool advertised, already scales with the surface. It is now
`None`; a deployment with a genuine limit can set it and get the old startup
warning back.

Measured on all 178 tools:

| Configuration | Bytes |
|---|---|
| auto-thinned | 153,199 |
| full schemas | 329,547 |

Nine more tools is not a question anyone needs to ask.

---

## 7. Sequence

**Phase 0 — prerequisites.** `daycount.py` and `contracts.py`. Neither is a
tool. Both are needed by everything else, and `year_fraction` has no
existing implementation to lean on. Fold the seven duplicated
`TRADING_DAYS = 252` literals in here while the file is being written.

**Phase I — the nine tools of §3.** No new dependency, no ML, no C++, no
provider work. Ships the runtime.

**Phase II — the desk tools.** `optimize_replication_basket`,
`analyze_etf_fair_value`, `price_total_return_swap`,
`analyze_total_return_future`, `analyze_dividend_points`,
`analyze_index_rebalance`. Takes the runtime to fifteen.

Two carry real cost. `optimize_replication_basket` needs a
tracking-error objective `(w−b)'Σ(w−b)` — `_solve_constrained`'s SLSQP
scaffolding and `_verify_solution` are directly reusable, but **max-names
cardinality has no precedent anywhere in the repo** and SLSQP cannot express
an integer constraint. Either add a QP/MIQP backend or wrap SLSQP in a
thresholding outer loop, and say which in the tool description.
`analyze_dividend_points` needs a dividend calendar the library cannot
fetch — take it as an argument.

**Phase III — infrastructure.** `data`: `build_continuous_futures_series`
(returning both a `research_series` and a `tradeable_contract_map`, because
a back-adjusted series is not a price anyone could have traded),
`fetch_contract_metadata`. `backtest`: `run_futures_backtest`, which is the
§1.6 state-model change — a `margin_posted` vector, variation margin to
cash, and an equity identity where a future contributes zero market value.
The C++ kernel should refuse the config via the existing `return None`
convention rather than be ported in lockstep.

**Phase IV — real time.** Not before the data layer has live futures, index
and ETF prices, which today it does not: Bloomberg has no intraday path and
no shipped provider exposes depth. Deferring this is the same call the
microstructure runtime made, and it was right — "twelve L2 tools with no L2
feed is twelve tools that refuse."

---

## 8. Where this leaves the library

```
                      DATA
                       │
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
  RESEARCH         DELTA ONE      MICROSTRUCTURE
  statistics       fair value        execution
  factors          basis             spread
  correlations     hedges            impact
      │                │                │
      └────────────────┼────────────────┘
                       ▼
                   PORTFOLIO
                       ▼
                    BACKTEST

  DERIVATIVES ──put-call parity──> implied forward ──> DELTA ONE
```

The three tools that change the library most, if only three were built:

**`analyze_cash_futures_basis`** — establishes the spot↔future relationship
the rest depends on.

**`size_futures_hedge`** — converts every regression already in the library
into a tradeable position.

**`compare_delta_one_expressions`** — the one that changes what the library
*is*. Not another collection of indicators: an instrument-equivalence
engine.
