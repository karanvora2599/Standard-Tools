# Correctness & Backend Parity

Both tiers of this package — the pure-Python/NumPy implementations and the
optional C++ extension — have been through line-by-line correctness audits.
This page is the record: what was found, what it broke, and what pins it now.

It is kept as a record rather than a summary because the pattern is the
useful part. Most findings were not exotic; they were ordinary code that
looked right, and the reason each survived review is usually more
instructive than the fix.
Most functions here have two implementations: a C++ kernel in `_sqt_core`
and a Python/NumPy fallback used when the extension isn't built. **The two
are contractually required to return the same answer**, and that requirement
is now tested directly rather than assumed.

Both tiers went through a line-by-line correctness audit (31 findings in the
Python tier, 10 in the C++ tier — full write-ups in
[CHANGELOG.md](../CHANGELOG.md)). Three themes are worth knowing as a user:

1. **Backend divergences.** Five cases were found where the same call
   returned a different answer depending on whether `_sqt_core` was built —
   `stochastic_oscillator` on a flat window, `cointegration_test`'s
   `autolag` handling, `hurst_exponent`'s regime post-processing,
   `rolling_factor_loadings` on an underdetermined window, and
   `profit_factor` when no trade wins or loses. All are fixed and pinned by
   tests that assert the two backends against *each other* — a test pinning
   only one side cannot see a divergence, which is exactly how several of
   these survived.

2. **Input validation is now uniform across tiers.** Non-finite inputs,
   invalid periods, and mismatched series lengths raise `ValidationError`
   at the Python boundary regardless of which tier executes. Previously
   several checks lived inside the C++ branch only, so the same bad input
   raised with the extension present and silently produced NaN without it.
   The most consequential case: `run_strategy(fill_price="next_open")` never
   validated `Open`, and because `cumprod` skips NaN, a gap there silently
   *dropped* that bar's P&L rather than surfacing — see
   [04_backtesting.md](04_backtesting.md#input-validation-contract).

3. **Memory safety in the Numba tier.** `@njit` compiles with bounds
   checking disabled. Two kernels (`_adx_numba`, `_psar_numba`) could write
   or read past their output arrays on short/empty input, returning
   plausible numbers instead of raising. Both are guarded, and all three
   execution tiers now agree on those inputs.

4. **Leakage in the modeling runtime.** A separate review of
   `standard_quant_tools.modeling` found seven critical issues the suite as
   it stood could not have caught, two of them look-ahead channels. Walk-
   forward validation was given only an integer `embargo`, never the target
   horizon, so training labels built from test-period prices survived the
   split — and the existing engine tests happened to pass
   `embargo == horizon`, which accidentally satisfied the missing invariant
   and hid it. Training rows are now purged by a per-row label end *date*:
   `horizon` counts an entity's own bars, so on a sparse calendar an integer
   offset under-purges exactly where it matters. Separately,
   `FeatureSpec.params` was unvalidated, and pandas reads a negative
   `pct_change` period as a *forward* window — so a negative lookback made a
   feature read future prices while its `pit_safe` label, and the
   point-in-time gate, stayed satisfied. Both are pinned by regression
   tests; see [15_modeling.md](15_modeling.md) and
   [CHANGELOG.md](../CHANGELOG.md) for the rest.

5. **The second modeling audit — 20 more items.** A follow-up review swept
   the modeling stack, the data layer beneath it, and the numerics both rest
   on. The unifying failure mode is a result that is plausible, internally
   consistent, and wrong:

   - A horizon-`h` label reads `Close[t+h]`, so a full-refit model has seen
     prices past its recorded `train_end_date`. Manifests now carry
     `training_information_cutoff` and `score_model` gates on it.
   - `end_date` was exclusive on yfinance and inclusive on Polygon and
     Bloomberg, so the default provider silently dropped the final bar. The
     ABC now specifies **inclusive** and all three providers trim to it.
   - `score_model` returned a "cross-section" that could mix dates, because
     each entity contributed its own latest surviving bar. Now one
     `effective_score_date`, with `stale_entities` and `staleness_days`.
   - A calendar gap in OOS predictions compressed the price axis: a
     boundary bar carried **26×** a normal daily return.
   - An alias could make a feature record *another* feature's implementation
     hash — the field whose whole job is answering what produced a column.
   - ICIR was computed as a mean of per-fold ICIRs, discarding exactly the
     between-fold variation it exists to measure.
   - Volatility features annualized with `sqrt(252)` at every interval.
   - The Python and C++ backtests disagreed on where a trade ends, so a
     resized position produced `num_trades=1` beside a two-row trade log.

   Every item was reproduced against a live interpreter before being fixed
   and is pinned by a regression test.

6. **The portfolio, screener and agent-tools audit — 10 more items.** The
   three packages the earlier passes had only touched incidentally. The
   sharpest two both produced a confident number that was not merely
   imprecise but inverted or fictional:

   - With observations ≤ assets a sample covariance is singular *by
     construction*, and the constrained optimizer answered by finding a
     direction in its null space — reporting a portfolio at ~0% volatility
     that carried **23% annualized volatility out of sample**.
   - `max_sharpe` returned the *minimum*-Sharpe portfolio whenever the
     risk-free rate reached the minimum-variance return, because
     normalizing the tangency solution by a negative sum flips it onto the
     inefficient branch.
   - A beta that could not be estimated was reported as `0.0`, so a ticker
     with no overlapping history **passed** a `beta_max` screen.
   - A NaN filter bound made an oversold screen a no-op that admitted
     RSI 100, since NaN fails every comparison.

   Both optimizer findings also split the two solver paths, which now share
   one gate.

7. **A full-codebase audit, Pass 1 — the older quant runtime.** A fresh
   review found the modeling runtime is no longer the weak point; the
   remaining risk sat in backtesting, metrics, data normalization and the
   audit trail, which never gained the input/output contracts modeling now
   enforces. The temporal and integrity findings are fixed:

   - **Deleting a model's `manifest.json` bypassed every integrity check.**
     It is the package's commit point, so removing it — strictly easier than
     forging a hash inside it — made a tampered `model.joblib`
     **deserialize** where it had previously been refused. `joblib.load`
     executes code from the file it is handed.
   - **A negative strategy lookback read future prices.** Pandas treats a
     negative `pct_change` period as a *forward* window, so
     `momentum_timeseries(lookback=-20)` computed bar 25's signal from bar
     45's price. Not one of the eight strategies validated a parameter; all
     now share one contract.
   - **A sparse signal panel deleted trading days**, distorting annualized
     volatility by **32×** on identical prices.
   - **Intraday bars from different exchanges looked simultaneous** —
     London 15:00 BST and New York 15:00 EDT are five hours apart and were
     indexed identically. Intraday is canonical UTC now.
   - **A corrupted audit trail silently restarted at genesis** instead of
     refusing to extend a damaged chain.
   - **"Unknown" stopped meaning free**: a ticker with no volume data used
     to score `$0` market impact against `$3bn` for one with real data.

8. **Pass 2 — one shared numerical contract.** Around 40 of the audit's
   findings were a single problem wearing different clothes:
   `@validate_series` checked emptiness and nothing else, so the same invalid
   input gave `nan` from one metric, `+inf` from another, and an
   `IndexError` from a third. Worst of all, `max_drawdown` on a series
   containing one infinity returned **-1.70** — a drawdown that looks
   measured. `standard_quant_tools.numeric_contract` now states the rules
   once: infinities and all-NaN are rejected everywhere, partial NaN is
   deliberately still allowed (warm-up windows are legitimate), prices must
   be strictly *positive* rather than merely finite, and `periods_per_year`
   is validated wherever it multiplies. Cost primitives no longer accept
   negative rates, which returned negative costs — a backtest paid to trade.

9. **Passes 3–5 — solvers, schemas and audit policy.** A solver reporting
   success is not a valid answer: a covariance with condition number
   **3.8e+14** (full rank, so the rank check passed) produced a maximum
   weight of **197,838× capital** with `converged: True`, and a long-only
   `target_return=99.0` returned tidy weights achieving **0.2443**. Returned
   weights are now checked against their own constraints. The classic agent
   schemas gained the Literals and bounds the modeling schemas already had —
   including `sort_by`, where an unrecognized metric had been **silently
   ignored**, and a combinatorial budget on `param_grid`. The audit trail
   gained a fail-closed mode, refuses to replay a redacted record (redaction
   and exact replay are in tension by construction), and treats a
   previously-failed call as a first-class replay outcome.

If you have audit records written before this release, note that
`content_hash` values are not comparable across the change — see the format
note in [10_auditability.md](10_auditability.md). The
tamper-evident record chain is unaffected and still verifies.

---
