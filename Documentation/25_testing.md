# The testing regime

Five layers, each catching a class of defect the others cannot. The
arrangement is deliberate: every layer exists because something got through
the ones above it.

| Layer | Where | What it catches | Runtime |
|---|---|---|---|
| Per-module correctness | `tests/<package>/` | Wrong answers | ~3 min |
| Whole-surface invariants | `tests/surface/test_invariants.py` | A tool registered halfway | 6 s |
| Adversarial fuzzing | `tests/surface/test_adversarial_inputs.py` | Unhandled exceptions, NaN in output | ~90 s |
| Determinism and purity | `tests/surface/test_determinism.py` | Ignored seeds, mutated arguments | ~7 min |
| Documentation | `tests/docs/` | Stale counts, undocumented tools, dead links | 6 s |
| Mutation testing | `Development/mutation_testing.py` | **Tests that would not notice** | ~15 min |

```bash
pytest                          # everything
pytest -m "not slow"            # skip the two heavy surface files
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
- Every runtime fits the per-runtime ceiling at the default detail, and
  thinning never leaves a schema unreachable.
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
the newest tools, where the bugs are, precisely the ones never fuzzed. 139
tools get a valid baseline and ten mutation families per numeric argument:
empty, single-element, all-identical, all-zero, NaN, infinity, 1e300,
1e-300, negated, truncated.

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

### Safety

The harness mutates source files on disk and restores them in a `finally`.
A `finally` does not run when the process is **killed**, and a ten-minute
CI timeout once left `if False:` in place of a bounds check.

So it refuses to start when a file it is about to mutate has uncommitted
changes, which makes `git checkout --` an unconditional recovery.
`--restore` does exactly that, and is what to run after an interrupted
session.

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
