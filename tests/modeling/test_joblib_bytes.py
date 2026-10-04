"""
`model.joblib` is the same bytes for the same model (see the CHANGELOG
entry of 2026-10-04).

A fresh fit already reproduced, but two things in the file did not follow
the model. A tree's node records are 64 bytes holding 57 bytes of fields,
and pickle writes all 64: a forest loaded from a file carries whatever its
loader left in the other seven, so two loads of one file re-dumped to two
hashes. And a histogram-boosting model pickles its bin mapper's
`n_threads`, the OpenMP thread count it was fitted under -- the physical
core count when nothing limits it -- so the same model gave different bytes
under limits of 1 and 4 threads, or on two machines.

Pinned here, through `serialization.save_joblib` (what the registry writes
`model.joblib` and `quantile_models.joblib` with): two identical fits, the
loads of one file, and a histogram-boosting model fitted under OpenMP limits
1 and 4 each give one file; a planted padding byte and a planted thread
count do not reach the file; no field and no prediction moves; the model in
memory keeps its thread count; and a registration writes the file this way.
A load re-dumped is not byte for byte the fitted model's file, for any
estimator, because pickle records which objects a model shares (see
`test_the_loads_of_one_file_give_one_file`).
"""

import copy
import io

import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from standard_quant_tools import _blas
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.registry import serialization
from standard_quant_tools.modeling.specs import (
    EstimatorSpec,
    ModelSpec,
    ValidationSpec,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec


def _data(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    y = X @ rng.standard_normal(5) + rng.standard_normal(n)
    return X, y


def _forest(X, y):
    return RandomForestRegressor(
        n_estimators=12, max_depth=6, random_state=0, n_jobs=1
    ).fit(X, y)


def _hgb(X, y):
    return HistGradientBoostingRegressor(max_iter=25, random_state=0).fit(X, y)


def _saved(tmp_path, obj, name):
    return open(serialization.save_joblib(tmp_path, name, obj), "rb").read()


def _trees(model):
    trees, _mappers = serialization._trees_and_bin_mappers(model)
    return trees


def _nodes(tree):
    return tree.__getstate__()["nodes"]


def _fields(model):
    """Every field of every tree's nodes, and the leaf values, without the
    padding: what the model predicts from."""
    out = []
    for tree in _trees(model):
        nodes = _nodes(tree)
        out.append([nodes[name].tobytes() for name in nodes.dtype.names])
        out.append(tree.__getstate__()["values"].tobytes())
    return out


def _planted_padding(model, value):
    """Set every padding byte of every node record to `value`: what an
    allocator that did not zero them leaves."""
    for tree in _trees(model):
        nodes = _nodes(tree)
        records = nodes.view("u1").reshape(nodes.size, nodes.dtype.itemsize)
        records[:, 57:] = value


class TestTheNodeRecords:
    def test_seven_bytes_of_a_node_record_belong_to_no_field(self):
        """The layout the zeroing relies on, read off scikit-learn's own
        dtype rather than assumed."""
        X, y = _data()
        nodes = _nodes(_trees(_forest(X, y))[0])
        assert nodes.dtype.itemsize == 64
        assert serialization._padding_runs(nodes.dtype) == ((57, 64),)

    def test_two_identical_fits_give_one_file(self, tmp_path):
        X, y = _data()
        for name, fit in (
            ("forest", _forest),
            (
                "gradient_boosting",
                lambda X, y: GradientBoostingRegressor(
                    n_estimators=15, max_depth=3, random_state=0
                ).fit(X, y),
            ),
            ("hgb", _hgb),
        ):
            first, second = fit(X, y), fit(X, y)
            assert _saved(tmp_path, first, f"{name}_a") == _saved(
                tmp_path, second, f"{name}_b"
            ), name

    def test_the_loads_of_one_file_give_one_file(self, tmp_path):
        """Each load of a forest leaves its own bytes in the padding of the
        node records it copies, and plain joblib wrote them: two loads of
        one file re-dumped to two files. Re-dumped here they are one, the
        trees' fields are the fit's, and each load predicts as the fit does.

        The re-dump is not the fit's own file, for any estimator: pickle
        writes an object referenced twice once, and a fit shares objects a
        load does not (a Ridge's coefficients and intercept share numpy's
        float64 dtype object; a forest's parameter names are the same
        strings as its trees' attribute names). That is pickle's record of
        object identity, not of the model."""
        X, y = _data()
        forest = _forest(X, y)
        written = _saved(tmp_path, forest, "fit")
        loads = [joblib.load(io.BytesIO(written)) for _ in range(3)]
        plain = []
        for loaded in loads:
            buffer = io.BytesIO()
            joblib.dump(loaded, buffer)
            plain.append(buffer.getvalue())
        assert len(set(plain)) > 1
        redumps = [_saved(tmp_path, m, f"load{i}") for i, m in enumerate(loads)]
        assert len(set(redumps)) == 1
        for loaded in loads:
            assert _fields(loaded) == _fields(forest)
            assert np.array_equal(loaded.predict(X), forest.predict(X))

    def test_a_planted_padding_byte_does_not_reach_the_file(self, tmp_path):
        """Null case beside it: the padding is the only thing that differs,
        and plain joblib writes it."""
        X, y = _data()
        clean, dirty = _forest(X, y), _forest(X, y)
        _planted_padding(dirty, 0xAB)
        plain = io.BytesIO()
        joblib.dump(dirty, plain)
        reference = io.BytesIO()
        joblib.dump(clean, reference)
        assert plain.getvalue() != reference.getvalue()
        fields = _fields(dirty)
        predictions = dirty.predict(X)
        assert _saved(tmp_path, dirty, "dirty") == _saved(tmp_path, clean, "clean")
        # The padding is cleared in the forest itself; no field moved.
        assert _fields(dirty) == fields
        assert np.array_equal(dirty.predict(X), predictions)
        for tree in _trees(dirty):
            nodes = _nodes(tree)
            records = nodes.view("u1").reshape(nodes.size, nodes.dtype.itemsize)
            assert not records[:, 57:].any()

    def test_a_gradient_boosting_model_s_trees_are_reached(self, tmp_path):
        """Its trees sit in an object array of shape (n_estimators, 1)."""
        X, y = _data()
        models = [
            GradientBoostingRegressor(n_estimators=10, random_state=0).fit(X, y)
            for _ in range(2)
        ]
        assert len(_trees(models[0])) == 10
        _planted_padding(models[1], 0x5C)
        assert _saved(tmp_path, models[0], "a") == _saved(tmp_path, models[1], "b")


class TestTheFitTimeThreadCount:
    def test_fitted_under_openmp_limits_1_and_4(self, tmp_path):
        """The same model fitted on one thread and on four writes one file,
        predicts the same, and keeps the count it was fitted under in
        memory; a model loaded from the file records None, scikit-learn's
        default for a bin mapper."""
        X, y = _data()
        models, recorded = {}, {}
        for limit in (1, 4):
            with _blas.openmp_thread_limit(limit):
                models[limit] = _hgb(X, y)
            recorded[limit] = models[limit]._bin_mapper.n_threads
        if recorded[1] == recorded[4]:
            pytest.skip("this machine runs both limits on one thread")
        one, four = models[1]._predictors, models[4]._predictors
        if len(one) != len(four) or not all(
            np.array_equal(a[0].nodes, b[0].nodes) for a, b in zip(one, four)
        ):
            # The bytes can only be one file when the trees are one model.
            pytest.skip("this OpenMP runtime fitted different trees at 1 and 4")
        written = {
            limit: _saved(tmp_path, model, f"hgb{limit}")
            for limit, model in models.items()
        }
        assert written[1] == written[4]
        assert {k: m._bin_mapper.n_threads for k, m in models.items()} == recorded
        loaded = joblib.load(io.BytesIO(written[1]))
        assert loaded._bin_mapper.n_threads is None
        for model in (models[4], loaded):
            assert np.array_equal(model.predict(X), models[1].predict(X))

    def test_a_planted_thread_count_does_not_reach_the_file(self, tmp_path):
        """Whatever the machine: one fit, two copies recording different
        counts."""
        X, y = _data()
        model = _hgb(X, y)
        copies = [copy.deepcopy(model) for _ in range(2)]
        copies[0]._bin_mapper.n_threads = 3
        copies[1]._bin_mapper.n_threads = 7
        assert _saved(tmp_path, copies[0], "a") == _saved(tmp_path, copies[1], "b")
        assert [c._bin_mapper.n_threads for c in copies] == [3, 7]

    def test_nested_bin_mappers_are_reached(self, tmp_path):
        """A calibrated classifier's per-fold models, and the quantile
        models' dict, which `quantile_models.joblib` holds."""
        X, y = _data()
        labels = (y > 0).astype(int)
        calibrated = CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_iter=10, random_state=0),
            cv=3,
            ensemble=True,
        ).fit(X, labels)
        quantiles = {
            q: HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=10, random_state=0
            ).fit(X, y)
            for q in (0.1, 0.9)
        }
        for name, obj in (("calibrated", calibrated), ("quantiles", quantiles)):
            mappers = serialization._trees_and_bin_mappers(obj)[1]
            assert len(mappers) == (3 if name == "calibrated" else 2)
            twin = copy.deepcopy(obj)
            for mapper in serialization._trees_and_bin_mappers(twin)[1]:
                mapper.n_threads = 5
            assert _saved(tmp_path, obj, f"{name}_a") == _saved(
                tmp_path, twin, f"{name}_b"
            )

    def test_restored_when_the_dump_fails(self, tmp_path, monkeypatch):
        X, y = _data()
        model = _hgb(X, y)
        model._bin_mapper.n_threads = 6

        def fail(*_args, **_kwargs):
            raise OSError("planted: the disk is full")

        monkeypatch.setattr(_artifacts, "save_joblib", fail)
        with pytest.raises(OSError, match="planted"):
            serialization.save_joblib(tmp_path, "model", model)
        assert model._bin_mapper.n_threads == 6


class TestARegistration:
    def test_writes_model_joblib_this_way(self, patched_multi_factory, monkeypatch):
        """The registry dumps `model.joblib` with every node padding byte
        zero and the bin mapper's thread count None, and the model it
        registered keeps its own count."""
        seen = []
        real = _artifacts.save_joblib

        def spy(directory, name, obj):
            trees, mappers = serialization._trees_and_bin_mappers(obj)
            padding = []
            for tree in trees:
                nodes = _nodes(tree)
                records = nodes.view("u1").reshape(nodes.size, nodes.dtype.itemsize)
                padding.append(bool(records[:, 57:].any()))
            seen.append(
                (name, len(trees), padding, [m.n_threads for m in mappers], mappers)
            )
            return real(directory, name, obj)

        monkeypatch.setattr(_artifacts, "save_joblib", spy)
        for estimator, params in (
            ("hist_gradient_boosting", {"max_iter": 10}),
            ("random_forest", {"n_estimators": 4, "max_depth": 3}),
        ):
            seen.clear()
            model_id = _train_a_model_with_spec(
                _dataset_spec(),
                dataset_id=f"ds_bytes_{estimator}",
                model_spec=ModelSpec(
                    task="regression",
                    estimator=EstimatorSpec(type=estimator, params=params),
                    validation=ValidationSpec(
                        train_window=150, test_window=30, embargo=5
                    ),
                    random_seed=1,
                ),
            )
            [(name, n_trees, padding, threads, mappers)] = [
                entry for entry in seen if entry[0] == "model"
            ]
            model = joblib.load(_artifacts.run_dir(model_id) / "model.joblib")
            if estimator == "hist_gradient_boosting":
                assert threads == [None]
                assert model._bin_mapper.n_threads is None
                # Put back on the registered model once written.
                assert mappers[0].n_threads is not None
            else:
                assert n_trees == 4 and not any(padding)
