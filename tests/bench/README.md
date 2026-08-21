# Performance harnesses

Not pytest tests -- they take minutes and measure wall-clock time, so they are
run by hand (or by a dedicated CI job), not as part of the suite.

    # per-kernel scaling, serial and parallel
    SQT_NUM_THREADS=1 python tests/bench/bench_kernels.py
    python tests/bench/bench_kernels.py

    # universe-scale: pair scan, portfolio simulation, panel transform, Monte Carlo
    python tests/bench/bench_universe.py

The baseline these produced on 2026-08-21 is recorded in
`Development/optimization_plan.md` section 2. That document is the reason these
exist: every figure in it comes from one of these two scripts, so a claim in the
plan can be re-checked rather than taken on trust.

`bench_universe.py` measures per-unit costs on a small universe and multiplies
out to 500/2,000 tickers. The multiplication is printed alongside the measured
unit cost so the extrapolation is visible and checkable, not baked in.
