# Performance harnesses

Not pytest tests -- they take minutes and measure wall-clock time, so they are
run by hand (or by a dedicated CI job), not as part of the suite.

    # per-kernel scaling, serial and parallel
    SQT_NUM_THREADS=1 python tests/bench/bench_kernels.py
    python tests/bench/bench_kernels.py

    # universe-scale: pair scan, portfolio simulation, panel transform, Monte Carlo
    python tests/bench/bench_universe.py

    # option chains: implied vol over 476 contracts, greeks on a 61-spot grid,
    # the zero-gamma search -- one call per contract against one per chain
    python tests/bench/bench_options.py

    # the modeling pipeline: IC, dataset build, walk-forward, estimators
    python tests/bench/bench_modeling.py
    python tests/bench/bench_modeling.py ic build      # or one section at a time

    # one build of the extension against another (OpenMP runtime, PGO,
    # clang-cl): the same kernels on the raw bindings, median and spread per
    # process
    python tests/bench/bench_build.py --json vcomp_1.json
    SQT_NUM_THREADS=1 python tests/bench/bench_build.py --json vcomp_serial_1.json

    # the training workload for a profile-guided build (run it against the
    # SQT_PGO_GENERATE=ON extension; see Documentation/30_build_guide.md)
    python tests/bench/pgo_training.py

The baseline these produced on 2026-08-21 is what
[Documentation/16_performance.md](../../Documentation/16_performance.md)
reports: every kernel and universe-scale figure there comes from
`bench_kernels.py` or `bench_universe.py`, so a published number can be
re-measured rather than taken on trust.

`bench_options.py` backs the option-chain table in the same document.

`bench_build.py` backs its build-variant section. A difference between two
builds is smaller than the difference between two processes of one build, so
that section runs it ten times per build and thread setting, interleaving the
builds, and reports the median of the per-process medians with their range.
Two builds can be measured side by side without rebuilding in between by
copying `src/standard_quant_tools/` (extension included) to two directories
and pointing `PYTHONPATH` at each: the copied C++ sources travel with the
copy, so the import-time source check still matches.

`bench_universe.py` measures per-unit costs on a small universe and multiplies
out to 500/2,000 tickers. The multiplication is printed alongside the measured
unit cost so the extrapolation is visible and checkable, not baked in.

`bench_modeling.py` backs the CHANGELOG's synthetic-panel modelling figures:
the IC kernel, dataset builds, engine runs and per-estimator runs. It patches `DataFactory` with a synthetic in-memory universe, so no measurement
includes network time. Its `build` section attributes time to feature
computation directly rather than A/B-ing whole builds: repeated on an ordinary
workstation, a whole-build A/B of the same change returned ratios from 0.62x to
1.39x — a spread wider than the effect being measured. When an end-to-end
comparison is that noisy, the honest move is to measure the part that changed,
and to say so.
