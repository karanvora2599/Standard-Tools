# The testing regime

Nine layers, each catching a class of defect the others cannot. The
arrangement is deliberate: every layer exists because something got through
the ones above it.

| Layer | Where | What it catches | Runtime |
|---|---|---|---|
| Per-module correctness | `tests/<package>/` | Wrong answers | ~3 min |
| Parity vs contract | `tests/modeling/test_native_metrics.py` | Two backends agreeing on the WRONG answer | 6 s |
| Whole-surface invariants | `tests/surface/test_invariants.py` | A tool registered halfway | 6 s |
| Adversarial fuzzing | `tests/surface/test_adversarial_inputs.py` | Unhandled exceptions, NaN in output | ~4.5 min |
| Metamorphic relations | `tests/surface/test_metamorphic.py` | Consistently-wrong answers | 4 s |
| Determinism and purity | `tests/surface/test_determinism.py` | Ignored seeds, mutated arguments | ~7 min |
| Documentation | `tests/docs/` | Stale counts, undocumented tools, dead links | 6 s |
| Mutation testing | `Development/mutation_testing.py` | **Tests that would not notice** | ~15 min |

```bash
pytest                          # everything
pytest -m "not slow"            # skips five files, not two: both heavy surface
                                # suites plus test_strategies and two cpp_bindings
pytest tests/surface -q         # the surface layers alone
python Development/mutation_testing.py
```

## Layer 1 — correctness, against planted answers

The rule for every test in `tests/<package>/`: **test against an answer
known by construction, never against a recorded output.**

A test asserting `vanna == 0.00397781` passes forever and catches nothing.
A test checking vanna against a central finite difference of delta catches
an algebra slip, because the analytic form and the numerical derivative
share no code and cannot agree by accident.

Worked examples of the pattern:

| What is checked | Against what |
|---|---|
| Second-order greeks | Central differences of the first-order greeks |
| Put-call parity | Prices the model itself produced — a model-free identity |
| Risk parity | Inverse-volatility, which is the closed form at zero correlation |
| Max diversification ratio | Exactly 1.0 when everything is perfectly correlated |
| Implementation shortfall | Four hand-computed components that must sum to the total |
| Volatility drag | σ²/2, matched to two basis points |
| Roll's spread | A spread planted in a simulated bid-ask bounce |
| Purged CV | Zero training labels touching their test set, counted directly |

**Every detector gets a null case.** A runs test that finds streaks must
also decline to find them in independent trades; a change-point detector
that always finds a break carries authority it has not earned. Where a
statistic claims a false-positive rate, that rate is *measured over many
draws* rather than asserted on one seed — a single draw cannot tell a
broken test from an unlucky one.

## Layer 1b — parity is not conformance

Every C++ kernel has a twin test that toggles the module's own `HAS_CPP`
and runs both paths on the same input. That catches a kernel drifting from
the Python it replaced, which is what it was built for, and it is
structurally incapable of catching the two of them being wrong together.

**It missed one for exactly that reason.** `standardize_by_date` zeroed a
whole date's cross-section whenever any single entity was missing —
reporting every PRESENT name as sitting exactly at the cross-sectional
mean, which `panel_stats.hpp` names as the specific fabricated observation
it must not produce. Both backends did it, agreed completely, and passed
parity on every run. A test even asserted the behaviour, under the name
`test_nan_poisons_its_whole_date`.

It was not undiscovered — `panel_stats.cpp` called it a wart and reproduced
it deliberately, on the reasoning that "in practice it never fires:
alignment drops NaN rows before the panel reaches the engine". Registering
an externally computed panel retired that premise, because such a panel
keeps its warm-up NaNs.

So a kernel now gets **two oracles, answering two different questions**:

| oracle | question | catches |
|---|---|---|
| the other backend, via `HAS_CPP` | do the two implementations agree? | drift between them |
| an independent one, written from the header | is the agreed answer right? | a shared misunderstanding |

`test_native_metrics.py::TestStandardizeByDate._contract` is the second
kind: it computes the documented behaviour from scratch rather than from
either implementation. `TestRankWithinDate` uses pandas as the parity
oracle and the header's NaN rule as the contract one, and
`TestPermutationNull` — where the two backends *cannot* agree, since a C++
shuffle is not numpy's PCG64 — asserts the analytic standard deviation of
the null instead, which is the property both are supposed to have.

**And each kernel ships a benchmark that can toggle the backend.** The
first native plan instructed readers to reproduce its speedups with
`tests/bench/bench_modeling.py`; `HAS_CPP` appears nowhere in
`tests/bench/`, so none of its figures could be re-derived from committed
code. `rank_by_date` and `permutation_null_ic` each carry a
`@pytest.mark.benchmark` test that times both paths back to back and
asserts a floor multiple, so the claim and the check are the same artifact.

## Layer 1c — the configuration that could not be run

Seventeen modules each decide `HAS_CPP` for themselves by probing
`_sqt_core`. That is the right design — a kernel added later falls back on
its own rather than all-or-nothing — but for a long time it meant the
**no-extension configuration could not be executed at all**. Every fallback
was reachable only by monkeypatching one module's flag inside one test, so
roughly half the C++-adjacent code had no end-to-end coverage.

```bash
SQT_DISABLE_NATIVE=1 pytest        # every kernel on its Python path
```

One name made unimportable flips all seventeen, because they all import the
same one, and each takes the `except ImportError` branch it already had. No
module needed changing.

**It found nothing wrong, which is the useful result.** 6,514 passed, 519
skipped, zero failures — the extra skips are the parity and benchmark tests
that `importorskip` the extension, correctly. The run takes 11:22 against
6:54, which is the compiled path's contribution measured at suite scale
rather than per kernel.

### Why this is a testing concern and not a packaging one

It is also what stops the codebase being surveyed wrongly. Instrument which
functions execute, on a machine where the extension is present, and every
fallback reports as dead. A reachability analysis of this package did exactly
that and recommended deleting two of them — `n_workers`'s `ProcessPoolExecutor`
and eight of the ten `@njit` kernels. Neither is dead code. Both are the
`else:` arm of `if HAS_CPP`, and deleting either would have removed the
faster path from the one configuration that has no other.

**A measurement of what EXECUTES is not a measurement of what is REACHABLE**,
and the gap between them is an entire install configuration. Anything that
counts calls to decide what is live has to be run twice.

`tests/test_fallback_configuration.py` holds the switch to that: it asserts
every listed module honours it, that the default is still native (a
regression there would invalidate every published performance figure), that
both paths compute the same answers, and — the drift guard — that the list of
native-aware modules in the test IS the set found by scanning `src/`. That
last one failed on its first run and was right to: the list I wrote by hand
was missing `indicators.panel` and `indicators.volatility`.

## Layer 2 — whole-surface invariants

Registering a new tool touches six places: a library function, an input
model, a result model, the runtime's `TOOL_DEFS` and `TOOL_DISPATCH`, the
facade's imports, and `agent.__all__`. Miss one and the tool half-exists.

No per-tool test can catch this, because **the tool somebody forgot to
register is the tool nobody wrote a test for**. So the properties are
checked over the whole surface:

- The runtimes **partition** the tools — a tool in two runtimes would make
  the boundary advisory, which is the one thing the runtime design exists
  to prevent.
- Advertised equals dispatchable, in every runtime.
- Every tool has an output schema. Losing the return annotation is a
  one-character change that silently stops the MCP server declaring
  structured output.
- Every input model sets `extra="forbid"`. Pydantic's default *drops* an
  unknown field, so a hallucinated argument runs on defaults while the
  caller believes it configured something.
- Every runtime reports a real schema cost, and thinning never leaves a
  schema unreachable. The cost is measured rather than capped: what a
  client can afford is a property of the client.
- A foreign tool is refused **by name**; a hallucinated one is told it does
  not exist. Those need different answers — telling a model to widen its
  scope for a tool that exists nowhere sends it looking for a flag that
  cannot help.

## Layer 3 — adversarial fuzzing

The contract is narrow and absolute. For **any** input, a tool either:

1. raises a `QuantError` naming what was wrong and what to change, or
2. returns a result that serializes to strict JSON — no NaN, no infinity.

Anything else is a defect. An `IndexError` from inside pandas crosses the
boundary naming no tool, no argument and no remedy, and reads to a caller
like a library bug rather than a request the data could not support.

**Inputs are synthesized from the schema**, not hand-written. A hand-written
fixture list covers the tools that existed when it was written — which makes
the newest tools, where the bugs are, precisely the ones never fuzzed. All
209 tools get a valid baseline and ten mutation families per numeric
argument: empty, single-element, all-identical, all-zero, NaN, infinity,
1e300, 1e-300, negated, truncated.

**A tool without a baseline is named, and that is the point.** For
most of this layer's life the collector wrapped synthesis in
`except Exception: continue`, so a tool the synthesizer could not build left
the fuzz set silently — the parametrization simply got smaller, which looks
exactly like a full one. Twenty-five tools were outside every check here:
`get_order_book_metrics`, all four tools taking a polymorphic data source,
six modeling tools. The floor guard that was supposed to notice asked for
100 synthesizable tools out of a surface that had 178, so there were 78
tools of headroom for the gap to grow in.

`EXPECTED_UNSYNTHESIZABLE` declares every absence with its reason, and is at
present EMPTY -- every one of the 209 tools is synthesizable, so every
one is fuzzed. Two guards hold it that way: an undeclared gap fails, and a declared gap
that has since been fixed also fails, so the list cannot become a place
exemptions accumulate. The floor is expressed against the live surface
rather than a constant, because a fixed number turns into slack the moment
the surface grows past it.

**Dates are a range, not a point.** Both ends of a synthesized window used
to resolve to the same day, so every windowed tool in the fuzz set was
handed a zero-length window and only ever exercised its empty-range path.
They now span 400 business days — enough for a 252-day lookback — which is
most of why this layer went from ~90 s to ~4.5 min, and the first run of it
found `detect_regimes` advertising a `seed` that could not affect its
output.

### What this found

Six bugs on its first run.

**Three `IndexError`s** — `get_technical_analysis`,
`get_advanced_indicators`, `get_position_size` — from
`.dropna().iloc[-1]` on an indicator whose window exceeded the data. A
20-period band over one bar is all-NaN; `dropna()` empties it and pandas
raises from inside its own indexer. Twenty-one call sites shared the
pattern. `validation.last_finite` replaces them all and reports bars in,
bars surviving, and that the window is the thing to shorten.

**Five overflow and division failures** in the derivatives tools at 1e300
and 1e-300. `exp(x)` overflows a float at about x = 710, and a `vol·√T`
denominator underflows to exactly zero; `_positive` admitted both. Now
bounded — and bounded on **magnitude**, because the first attempt used a
signed range and broke the Bachelier exemption, which is the exact case
that model exists for. The existing tests caught that in the same run.

## Layer 3b — metamorphic relations

The correctness tests check one input against one known answer. The fuzzer
checks that nothing crashes. **Neither catches a function that is
consistently wrong** — one returning a plausible number for every input,
including the inputs where a known relation says it must return the *same*
number as some other input.

Metamorphic testing asks a different question: not "is this answer right"
but "do these two answers stand in the relation they must". It needs no
known answer, which is why it reaches code where the right answer is hard
to compute independently.

| Relation | Example |
|---|---|
| **Scale invariance** | Scaling a covariance matrix by 1e6 moves no risk-parity weight; `Sharpe(c·r) = Sharpe(r)` |
| **Order invariance** | Total return and the moments cannot depend on the order the returns arrived |
| **Order dependence** | ...and the runs count and the drawdown *must*, or they are measuring nothing |
| **Permutation equivariance** | Relabelling the assets permutes the weights and changes nothing else |
| **Monotonicity** | More observations narrow the interval; falling correlation raises the diversification ratio |
| **Translation** | Adding a constant moves the mean by that constant and leaves the variance alone |

Two of these catch bugs the other layers structurally cannot. **Scale
invariance** breaks the moment an absolute threshold creeps into a solver —
a comparison against `1e-8` that should have been relative — and a
single-input test would never see it. **Permutation equivariance** breaks
when something indexes by position where it should index by name, which
produces a plausible number every single time.

## Layer 4 — determinism and purity

Three properties, and the second is the one people forget.

- **Reproducible.** The same arguments give byte-identical output. Without
  it an audit record cannot be replayed.
- **Seed-sensitive.** A different seed gives a different answer. *A tool
  that accepts a seed and ignores it passes every reproducibility test ever
  written* — its output is perfectly stable — and a caller who varies the
  seed to check robustness gets the same number back and concludes the
  result is robust.
- **Pure.** Calling a tool does not mutate its arguments. A tool that sorts
  its input in place changes the caller's data, and the second iteration of
  a loop runs on something the caller never passed.

Scoped to **offline** tools. A tool taking a `symbol` does not have its
output determined by its arguments — the market is the other input, and it
moves between two calls. Reproducibility for those is a different property,
and the audit trail's `verify_replay` is what tests it, because that pins
the data as well as the arguments. Timestamp fields are stripped before
comparison: a provenance record without a time is not a provenance record.

## Layer 5 — documentation

Documentation rots in a way code does not: **nothing fails when it goes
stale**. Before `tests/docs/` existed, 85 of 157 tools appeared in no
document, the README advertised 95 tools across six runtimes, and three
guides quoted a count that had been wrong for months.

- `Documentation/20_tool_index.md` is **generated** from the live registry.
  A test regenerates it and compares byte-for-byte, so adding a tool without
  regenerating fails in the commit that added it.
- Every count appearing in prose is checked against reality — whole-surface
  counts, per-runtime counts quoted in `sqt-mcp` commands, the rows of the
  runtime table.
- Every internal link resolves. A link to a renamed file is a dead end a
  reader hits and an author never does.
- No tool description is a bare restatement of its own name, and none
  contains a newline (which renders unpredictably across function-calling
  clients).

## Layer 6 — mutation testing

**A passing suite proves the tests run. It does not prove they would notice
a defect.** Those are different claims, and only this layer distinguishes
them.

`Development/mutation_testing.py` holds a catalogue of deliberate defects —
a correction dropped as redundant, a sign flipped, a guard removed, a
permutation swapped for a resample — and reruns the tests that should care.
A mutation that **survives** marks a test that is decorative.

The mutations are deliberate rather than random. Random character damage
would be caught by any test and would prove nothing; each of these is a
change somebody could plausibly make while tidying.

### What this found

Three survived the first run, and two were real:

- The Granger false-positive test asserted `rate < 0.15`, and the
  **uncorrected** rate is 12–15%. It slipped under the very bar it existed
  to enforce.
- The seasonality test asserted `p_value_corrected >= p_value_raw`, which
  stays true when the mutation changes only the **flag**. The corrected
  number is still reported; it just stops being the one that decides.

Both are now pinned by a test that searches for a case where the correction
changes the *answer* — raw below 0.05, corrected above it — and asserts the
flag follows the corrected one. Neither can pass with the correction gone.

**Current state: 21 mutations, 21 killed, 0 survived.**

### Safety — and the incident that shaped it

The harness mutates source files on disk and restores them in a `finally`.
A `finally` does not run when the process is **killed**, and this went
wrong twice in one session:

1. A ten-minute timeout killed a run mid-mutation, leaving `if False:` in
   place of a bounds check in `analysis/options.py`. Caught by
   `git status`, restored with `git checkout --`.
2. The harness was then run in the background — and a `git add -A` for an
   unrelated commit swept the working tree **while a mutation was live**.
   `if False:` reached a commit, and the file then looked *clean* to git,
   because the mutation had become the committed state.

The second is the dangerous one. Nothing about the repository looked wrong
afterwards, and the disabled guard sat in the source with a full docstring
explaining why it was necessary.

Three defences, in order of how much they cover:

- **`require_clean_tree`** refuses to start when a file about to be mutated
  has uncommitted changes. This protects *your* work from the harness, and
  makes `git checkout --` an unconditional recovery.
- **`.mutation_active`** is written for the duration of a run. It cannot
  stop another process committing — nothing can — but it makes the state
  diagnosable afterwards.
- **`test_no_constant_condition_appears_in_the_source`** is the one that
  actually closes it. The constants the catalogue substitutes (`if False:`,
  `if True:`, `if 0:`, `if 1:`) cannot appear in committed source at all,
  so an escaped mutation fails the ordinary test run. It is a worthwhile
  check independently: `if False:` in committed code is either dead code or
  a disabled guard, and neither should survive review.

A fourth test, `test_the_mutation_catalogue_anchors_all_still_match`, fails
in the normal run when an anchor has drifted. A mutation whose anchor no
longer matches reports as SKIPPED rather than survived, which is easy to
read past — and a catalogue that has quietly stopped covering the code it
names is worse than no catalogue, because the report still says zero
survivors.

```bash
python Development/mutation_testing.py --list
python Development/mutation_testing.py --filter granger
python Development/mutation_testing.py --restore   # after an interruption
```

## Adding a tool: what the regime expects

1. The computation in a library module, with its **limits in the docstring**
   — how it fails, not only what it does.
2. An `Input` model with `ConfigDict(extra="forbid")` and a **typed** result
   model. An untyped return drops the MCP output schema and a test pins it.
3. Entries in the runtime's `TOOL_DEFS` and `TOOL_DISPATCH`, built from one
   list so a tool cannot be advertised without being dispatchable.
4. Re-exports from `agent/tools.py` and `agent/__init__.py.__all__`.
5. Tests against a **planted** answer, with a null case.
6. `python Development/generate_tool_index.py`, committed.

Layers 2–5 then cover the tool automatically — the fuzzer, the determinism
checks and the documentation tests all read the registry, so a tool added
today is fuzzed today.

## What none of this establishes

That the numbers are *useful*. The regime checks that a statistic is
computed correctly, refuses cleanly, reproduces exactly and is documented
honestly. Whether the strategy makes money is not a property any test file
can assert, and [24_overfitting.md](24_overfitting.md) is about how little
a backtest establishes on its own.

## Related

- [17_correctness.md](17_correctness.md) — the C++/Python backend-parity contract
- [10_auditability.md](10_auditability.md) — replay verification against recorded data
- [24_overfitting.md](24_overfitting.md) — the same scepticism applied to results
