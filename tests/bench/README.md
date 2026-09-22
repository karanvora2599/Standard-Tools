# Performance harnesses

Not pytest tests -- they take minutes and measure wall-clock time, so they are
run by hand (or by a dedicated CI job), not as part of the suite.

    # per-kernel scaling, serial and parallel
    SQT_NUM_THREADS=1 python tests/bench/bench_kernels.py
    python tests/bench/bench_kernels.py

    # universe-scale: pair scan, portfolio simulation, panel transform, Monte Carlo
    python tests/bench/bench_universe.py

    # the modeling pipeline: IC, dataset build, walk-forward, estimators
    python tests/bench/bench_modeling.py
    python tests/bench/bench_modeling.py ic build      # or one section at a time

The baseline these produced on 2026-08-21 is what
[Documentation/16_performance.md](../../Documentation/16_performance.md)
reports: every kernel and universe-scale figure there comes from
`bench_kernels.py` or `bench_universe.py`, so a published number can be
re-measured rather than taken on trust.

`bench_universe.py` measures per-unit costs on a small universe and multiplies
out to 500/2,000 tickers. The multiplication is printed alongside the measured
unit cost so the extrapolation is visible and checkable, not baked in.

`bench_modeling.py` backs every modelling figure in the CHANGELOG.
It patches `DataFactory` with a synthetic in-memory universe, so no measurement
includes network time. Its `build` section attributes time to feature
computation directly rather than A/B-ing whole builds: repeated on an ordinary
workstation, a whole-build A/B of the same change returned ratios from 0.62x to
1.39x — a spread wider than the effect being measured. When an end-to-end
comparison is that noisy, the honest move is to measure the part that changed,
and to say so.
