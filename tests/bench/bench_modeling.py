"""
Where the modeling pipeline spends its time.

Every figure quoted in Development/modeling_analysis.md comes from this
script, so a claim there can be re-checked rather than taken on trust. It
builds a synthetic OHLCV universe in memory and patches DataFactory, so no
measurement includes network time and the numbers are reproducible.

    python tests/bench/bench_modeling.py            # everything
    python tests/bench/bench_modeling.py ic         # one section
    python tests/bench/bench_modeling.py build engine

Sections:
    ic        cross_sectional_ic, vectorized vs the per-date groupby
    build     build_dataset scaling, and the panel feature fast path
    engine    run_experiment by validation scheme and preprocessing
    estimator the cost of each estimator on one walk-forward run

`estimator` is slow on purpose: random_forest is the thing being measured,
and it takes about a minute.
"""

import gc
import os
import sys
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd

_TMP = tempfile.mkdtemp()
os.environ.setdefault("SQT_RUNS_DIR", os.path.join(_TMP, "runs"))
os.environ.setdefault("SQT_AUDIT_DIR", os.path.join(_TMP, "audit"))
os.environ.setdefault("SQT_AUDIT_ENABLED", "0")

from standard_quant_tools.data.factory import DataFactory  # noqa: E402
from standard_quant_tools.data.metadata import DataSetMetadata  # noqa: E402
from standard_quant_tools.modeling.dataset import (  # noqa: E402
    builder as builder_module,
)
from standard_quant_tools.modeling.dataset.builder import build_dataset  # noqa: E402
from standard_quant_tools.modeling.engine import run_experiment  # noqa: E402
from standard_quant_tools.modeling.specs import (  # noqa: E402
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    PreprocessingSpec,
    SearchSpec,
    TargetSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation.metrics import (  # noqa: E402
    cross_sectional_ic,
)

N_BARS = 1500

# The five features build_dataset can route through the native panel call.
PANEL_FEATURES = [
    "technical.rsi",
    "technical.adx",
    "technical.stochastic_k",
    "risk.atr_pct",
    "risk.bollinger_pct_b",
]
MIXED_FEATURES = [
    "technical.rsi",
    "technical.adx",
    "technical.macd_histogram",
    "risk.atr_pct",
    "risk.realized_volatility",
    "market.momentum",
]


def _ohlcv(symbol: str, n: int) -> pd.DataFrame:
    # Seeded from the symbol so a rerun measures the same data.
    seed = abs(hash(symbol)) % 100_000
    rng = np.random.default_rng(seed)
    index = pd.date_range("2015-01-02", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.014, n)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.001, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.006, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
            "Close": close,
            "Volume": rng.integers(1e6, 5e6, n).astype(float),
        },
        index=index,
    )


def install_provider(n_bars: int = N_BARS) -> None:
    cache: dict = {}

    def fetch(symbol, start=None, end=None, *args, **kwargs):
        if symbol not in cache:
            cache[symbol] = _ohlcv(symbol, n_bars)
        return cache[symbol]

    provider = MagicMock()
    provider.get_ohlcv.side_effect = fetch
    # build_dataset fetches the universe through the ASYNC path; wiring only
    # get_ohlcv leaves every symbol failing with a TypeError on await.
    provider.get_ohlcv_async = AsyncMock(side_effect=fetch)
    provider.get_metadata.side_effect = lambda symbol, interval="1d": DataSetMetadata(
        provider="bench",
        adjusted=True,
        survivorship_free=False,
        point_in_time=False,
        frequency=interval,
        timezone="America/New_York",
    )
    DataFactory.get_provider = staticmethod(lambda *a, **kw: provider)  # type: ignore


def best_of(fn, reps: int = 3) -> float:
    fn()
    best = float("inf")
    gc.disable()
    try:
        for _ in range(reps):
            start = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - start)
    finally:
        gc.enable()
    return best


def dataset_spec(universe, features, target_type="forward_return"):
    return DatasetSpec(
        universe=universe,
        start="2015-01-02",
        end="2030-01-01",
        features=[FeatureSpec(id=f) for f in features],
        target=TargetSpec(type=target_type, horizon=5),
        benchmark="SPY",
    )


def model_spec(estimator="ridge", params=None, **kwargs):
    kwargs.setdefault(
        "validation", ValidationSpec(train_window=250, test_window=125, embargo=5)
    )
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type=estimator, params=params or {}),
        random_seed=1,
        **kwargs,
    )


def _header(title: str) -> None:
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


# ── ic ───────────────────────────────────────────────────────────────────
def _reference_ic(y_true, y_pred, dates, method):
    """The per-date implementation cross_sectional_ic replaced."""
    frame = pd.DataFrame({"date": dates, "y": y_true, "p": y_pred})
    per_date = {}
    for date, group in frame.groupby("date", sort=True):
        if len(group) < 2:
            continue
        value = float(group["y"].corr(group["p"], method=method))
        per_date[date] = 0.0 if np.isnan(value) else value
    return pd.Series(per_date, dtype=float)


def bench_ic() -> None:
    _header("cross_sectional_ic - vectorized vs the per-date groupby it replaced")
    print(
        f"{'dates':>6} {'entities':>9} {'rows':>10} {'before':>10} {'after':>10} "
        f"{'speedup':>9} {'worst |diff|':>14}"
    )
    rng = np.random.default_rng(0)
    for n_dates, n_entities in (
        (63, 50),
        (252, 50),
        (252, 500),
        (1000, 500),
        (252, 2000),
    ):
        dates = np.repeat(pd.date_range("2020-01-01", periods=n_dates), n_entities)
        y = rng.normal(0, 1, n_dates * n_entities)
        p = 0.3 * y + rng.normal(0, 1, n_dates * n_entities)
        before = best_of(lambda: _reference_ic(y, p, dates, "spearman"), 2)
        after = best_of(lambda: cross_sectional_ic(y, p, dates, "spearman"), 5)
        diff = float(
            np.max(
                np.abs(
                    _reference_ic(y, p, dates, "spearman").to_numpy()
                    - cross_sectional_ic(y, p, dates, "spearman").to_numpy()
                )
            )
        )
        print(
            f"{n_dates:>6} {n_entities:>9} {n_dates * n_entities:>10,} "
            f"{before * 1e3:>9.1f}m {after * 1e3:>9.2f}m {before / after:>8.1f}x "
            f"{diff:>14.3e}"
        )
    print()
    print("  The multiple shrinks as the cross-section grows: at 500 entities each")
    print("  date carries enough real work that the per-date overhead amortizes.")


# ── build ────────────────────────────────────────────────────────────────
def bench_build() -> None:
    _header("build_dataset - scaling, and the native panel feature fast path")
    install_provider()
    print(
        f"{'entities':>9} {'features':>9} {'time':>10} {'per entity':>12} {'rows':>10}"
    )
    for n_entities in (5, 20, 50, 100):
        universe = [f"S{i:04d}" for i in range(n_entities)]
        spec = dataset_spec(universe, MIXED_FEATURES)
        elapsed = best_of(lambda: build_dataset(spec), 2)
        rows = len(build_dataset(spec)["panel"])
        print(
            f"{n_entities:>9} {len(MIXED_FEATURES):>9} {elapsed * 1e3:>9.1f}m "
            f"{elapsed / n_entities * 1e3:>11.2f}m {rows:>10,}"
        )

    print()
    print("  panel fast path - time attributed to FEATURE COMPUTATION only")
    print(f"{'entities':>9} {'per-entity':>12} {'panel':>10} {'speedup':>9}")
    # Measured by instrumenting the feature callables and the panel call
    # directly, rather than by A/B-ing whole builds. That is deliberate:
    # a whole-build A/B of this change is dominated by run-to-run noise on
    # an ordinary workstation -- repeated here, the same comparison returned
    # ratios from 0.62x to 1.39x, a spread wider than the effect. Attributing
    # the time directly measures the thing that actually changed.
    import standard_quant_tools.modeling.features.registry as feature_registry

    feature_time = {"total": 0.0}
    for definition in feature_registry.FEATURE_REGISTRY.values():
        if getattr(definition.fn, "_bench_timed", False):
            continue

        def _wrap(fn):
            def timed(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    feature_time["total"] += time.perf_counter() - start

            timed._bench_timed = True
            return timed

        definition.fn = _wrap(definition.fn)

    real = builder_module.compute_panel_features
    panel_time = {"total": 0.0}

    def timed_panel(*args, **kwargs):
        start = time.perf_counter()
        try:
            return real(*args, **kwargs)
        finally:
            panel_time["total"] += time.perf_counter() - start

    for n_entities in (20, 50, 100):
        universe = [f"S{i:04d}" for i in range(n_entities)]
        spec = dataset_spec(universe, PANEL_FEATURES)
        builder_module.compute_panel_features = lambda *a, **k: {}
        build_dataset(spec)
        feature_time["total"] = 0.0
        build_dataset(spec)
        per_entity = feature_time["total"]

        builder_module.compute_panel_features = timed_panel
        build_dataset(spec)
        panel_time["total"] = 0.0
        build_dataset(spec)
        panel = panel_time["total"]
        builder_module.compute_panel_features = real

        print(
            f"{n_entities:>9} {per_entity * 1e3:>11.1f}m {panel * 1e3:>9.1f}m "
            f"{per_entity / panel:>8.2f}x"
        )
    print()
    print("  Whole-build effect is smaller than this, because feature computation")
    print("  is only part of a build -- fetching, stacking, alignment and hashing")
    print("  are unchanged. Expect roughly break-even at 20 entities and 1.1-1.4x")
    print("  at 50-100, with the measurement noisier than the effect at the low end.")


# ── engine ───────────────────────────────────────────────────────────────
def bench_engine() -> None:
    _header("run_experiment - validation scheme, preprocessing, search")
    install_provider()
    universe = [f"S{i:04d}" for i in range(50)]
    dataset = build_dataset(dataset_spec(universe, MIXED_FEATURES))
    rows = len(dataset["panel"])
    print(f"  panel: {rows:,} rows, {len(universe)} entities")
    print()
    print(f"{'configuration':<44} {'folds':>6} {'time':>10} {'ms/fold':>9}")

    configurations = [
        ("walk_forward rolling (default)", model_spec()),
        (
            "walk_forward expanding",
            model_spec(
                validation=ValidationSpec(
                    train_window=250, test_window=125, embargo=5, scheme="expanding"
                )
            ),
        ),
        (
            "purged_kfold n_splits=5",
            model_spec(
                validation=ValidationSpec(method="purged_kfold", n_splits=5, embargo=5)
            ),
        ),
        (
            "cross-sectional normalization",
            model_spec(
                preprocessing=PreprocessingSpec(normalization="cross_sectional")
            ),
        ),
        (
            "grid search, 4 alphas x 3 inner folds",
            model_spec(
                search=SearchSpec(
                    param_grid={"alpha": [0.01, 1.0, 100.0, 10000.0]}, inner_splits=3
                )
            ),
        ),
    ]
    for label, spec in configurations:
        elapsed = best_of(lambda: run_experiment(dataset, spec, "bench"), 2)
        folds = run_experiment(dataset, spec, "bench")["n_folds"]
        print(
            f"{label:<44} {folds:>6} {elapsed * 1e3:>9.1f}m "
            f"{elapsed / max(folds, 1) * 1e3:>8.1f}"
        )


# ── estimator ────────────────────────────────────────────────────────────
def bench_estimator() -> None:
    _header("run_experiment - cost by estimator (one walk-forward run)")
    install_provider()
    universe = [f"S{i:04d}" for i in range(50)]
    dataset = build_dataset(dataset_spec(universe, MIXED_FEATURES))
    from standard_quant_tools.modeling.estimators import boosting

    candidates = [
        ("linear", {}),
        ("ridge", {"alpha": 1.0}),
        ("hist_gradient_boosting", {}),
        ("random_forest", {"n_estimators": 100}),
    ]
    if boosting.HAS_LIGHTGBM:
        candidates.append(("lightgbm", {"n_estimators": 100}))
    if boosting.HAS_XGBOOST:
        candidates.append(("xgboost", {"n_estimators": 100}))

    print(f"{'estimator':<28} {'folds':>6} {'time':>11} {'s/fold':>9}")
    for name, params in candidates:
        spec = model_spec(name, params)
        gc.disable()
        try:
            start = time.perf_counter()
            result = run_experiment(dataset, spec, "bench")
            elapsed = time.perf_counter() - start
        finally:
            gc.enable()
        folds = result["n_folds"]
        print(
            f"{name:<28} {folds:>6} {elapsed:>10.2f}s "
            f"{elapsed / max(folds, 1):>8.3f}"
        )
    print()
    print("  random_forest is the reason lightgbm/xgboost were added: it is the")
    print("  estimator that stops being usable first as the universe grows.")


SECTIONS = {
    "ic": bench_ic,
    "build": bench_build,
    "engine": bench_engine,
    "estimator": bench_estimator,
}


def main() -> None:
    requested = [a for a in sys.argv[1:] if not a.startswith("-")]
    unknown = [name for name in requested if name not in SECTIONS]
    if unknown:
        raise SystemExit(
            f"unknown section(s) {unknown}; choose from {sorted(SECTIONS)}"
        )
    for name in requested or list(SECTIONS):
        SECTIONS[name]()


if __name__ == "__main__":
    main()
