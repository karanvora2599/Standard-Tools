"""
The skops bundle: the registered estimator, loadable without pickle.

joblib is pickle, and pickle executes code from the file. The bundle is
the same estimator written as declared state, and the loader constructs
only the types it was told to trust. Planted: a tampered joblib is
refused while the bundle still loads, a tampered bundle is refused before
it is read, and a bundle that names a type from outside this package is
refused by that type's name.
"""

import io
import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling import engine
from standard_quant_tools.modeling.capabilities import modeling_capabilities
from standard_quant_tools.modeling.estimators.registry import ESTIMATOR_REGISTRY
from standard_quant_tools.modeling.estimators.survival import CoxPHRegressor
from standard_quant_tools.modeling.registry.model_registry import (
    load_manifest,
    load_model,
    load_monitoring_reference,
)
from standard_quant_tools.modeling.registry.serialization import (
    FORMAT_ENV,
    TRUSTED_PREFIX,
    TRUSTED_TYPES,
    deterministic_archive,
    dump_estimator,
    load_estimator,
    skops_available,
    untrusted_types,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec
from .test_survival import _planted

needs_skops = pytest.mark.skipif(not skops_available(), reason="skops is not installed")


class _NotOurs:
    """A type this registry never writes."""

    def __init__(self):
        self.value = 1


def _features(model_id):
    manifest = load_manifest(model_id)
    _profile, reference, _predictions = load_monitoring_reference(model_id)
    return reference[manifest.feature_ids].to_numpy(dtype=float)


@needs_skops
class TestTheBundle:
    def test_registration_writes_a_hashed_bundle_that_loads_the_same_model(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_skops")
        manifest = load_manifest(model_id)
        assert manifest.formats == ["joblib", "skops"]
        assert "model.skops" in manifest.content_hashes
        assert (_artifacts.run_dir(model_id) / "model.skops").exists()
        X = _features(model_id)
        via_joblib = load_model(model_id).predict(X)
        via_skops = load_model(model_id, format="skops").predict(X)
        assert np.allclose(via_joblib, via_skops)

    def test_the_environment_chooses_the_format(
        self, patched_multi_factory, monkeypatch
    ):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_skops_env")
        X = _features(model_id)
        expected = load_model(model_id).predict(X)
        # Corrupt the joblib: the default load refuses, the bundle still answers.
        with open(_artifacts.run_dir(model_id) / "model.joblib", "ab") as handle:
            handle.write(b"\x00")
        with pytest.raises(ValidationError, match="changed since it was registered"):
            load_model(model_id)
        monkeypatch.setenv(FORMAT_ENV, "skops")
        assert np.allclose(load_model(model_id).predict(X), expected)
        monkeypatch.setenv(FORMAT_ENV, "onnx")
        with pytest.raises(ValidationError, match="not a model format"):
            load_model(model_id)
        with pytest.raises(ValidationError, match="not one of"):
            load_model(model_id, format="pickle")

    def test_a_tampered_bundle_is_refused_before_it_is_read(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_skops_tamper"
        )
        with open(_artifacts.run_dir(model_id) / "model.skops", "ab") as handle:
            handle.write(b"\x00")
        with pytest.raises(ValidationError, match="model.skops has changed"):
            load_model(model_id, format="skops")

    def test_a_foreign_type_is_refused_by_name(self, tmp_path):
        import skops.io as sio

        path = tmp_path / "foreign.skops"
        sio.dump(_NotOurs(), path)
        with pytest.raises(ValidationError, match="_NotOurs"):
            load_estimator(str(path))

    def test_this_package_s_own_estimators_round_trip(self, tmp_path):
        X, duration, event = _planted(200, seed=4)
        model = CoxPHRegressor(alpha=0.5).fit(X, np.column_stack([duration, event]))
        path = dump_estimator(tmp_path, "cox", model)
        assert path is not None and path.endswith("cox.skops")
        restored = load_estimator(path)
        assert np.allclose(restored.predict(X), model.predict(X))
        assert np.allclose(restored.baseline_cumhaz_, model.baseline_cumhaz_)

    def test_a_model_without_a_bundle_refuses_the_format_by_name(
        self, patched_multi_factory, monkeypatch
    ):
        import skops.io as sio

        def _cannot(*_args, **_kwargs):
            raise TypeError("planted: this estimator holds a native handle")

        monkeypatch.setattr(sio, "dump", _cannot)
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_skops_none")
        manifest = load_manifest(model_id)
        assert manifest.formats == ["joblib"]
        assert "model.skops" not in manifest.content_hashes
        with pytest.raises(ValidationError, match="no skops bundle"):
            load_model(model_id, format="skops")
        assert load_model(model_id) is not None

    def test_the_capability_report_says_whether_bundles_are_written(self):
        assert modeling_capabilities()["optional_dependencies"]["skops"] is True


def _fitted(seed: int = 0):
    """The same fits every call: a ridge, a forest, a pipeline and this
    package's Cox model, plus a ridge carrying a bytes value and a second
    reference to one of its arrays -- what skops stores under a random name
    and as a reference to an object already written."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((300, 5))
    y = X @ rng.standard_normal(5) + 0.1 * rng.standard_normal(300)
    carrying = Ridge(alpha=0.5).fit(X, y)
    carrying.note_ = b"\x00planted bytes"
    carrying.coef_again_ = carrying.coef_
    X_cox, duration, event = _planted(200, seed=4)
    return X, {
        "ridge": Ridge(alpha=1.0).fit(X, y),
        "forest": RandomForestRegressor(
            n_estimators=4, max_depth=4, random_state=0, n_jobs=1
        ).fit(X, y),
        "pipeline": make_pipeline(StandardScaler(), Ridge()).fit(X, y),
        "carrying": carrying,
        "cox": (
            CoxPHRegressor(alpha=0.5).fit(X_cox, np.column_stack([duration, event])),
            X_cox,
        ),
    }


#: A tree node record's layout in miniature: three fields covering 17 of 24
#: bytes, so seven bytes per record belong to no field.
_NODE = np.dtype(
    {
        "names": ["left", "threshold", "flag"],
        "formats": ["<i8", "<f8", "u1"],
        "offsets": [0, 8, 16],
        "itemsize": 24,
    }
)


def _nodes_npy(padding: int) -> bytes:
    """Three node records saved as skops saves an array, every padding
    byte set to `padding`: what an allocator that did not zero them left."""
    nodes = np.zeros(3, dtype=_NODE)
    nodes["left"] = [1, -1, -1]
    nodes["threshold"] = [0.5, -2.0, -2.0]
    nodes["flag"] = [1, 0, 0]
    raw = bytearray(nodes.tobytes())
    for record in range(3):
        raw[record * 24 + 17 : record * 24 + 24] = bytes([padding]) * 7
    buffer = io.BytesIO()
    np.save(buffer, np.frombuffer(bytes(raw), dtype=_NODE), allow_pickle=False)
    return buffer.getvalue()


def _skops_shaped(
    address: int,
    other: int,
    uuid_name: str,
    stamp,
    array: bytes = b"\x93NUMPY planted array",
) -> bytes:
    """An archive laid out the way `skops.io.dump` lays one out: an array
    stored under its object's address, a second reference to it, a bytes
    value under a UUID, `schema.json` last, every member dated `stamp`."""
    schema = {
        "__class__": "Ridge",
        "__module__": "sklearn.linear_model._ridge",
        "__loader__": "ObjectNode",
        "content": {
            "__class__": "dict",
            "__module__": "builtins",
            "__loader__": "DictNode",
            "content": {
                "coef_": {
                    "__class__": "ndarray",
                    "__module__": "numpy",
                    "__loader__": "NdArrayNode",
                    "type": "numpy",
                    "file": f"{address}.npy",
                    "__id__": address,
                },
                "coef_again_": {
                    "__class__": "ndarray",
                    "__module__": "numpy",
                    "__loader__": "CachedNode",
                    "__id__": address,
                },
                "note_": {
                    "__class__": "bytes",
                    "__module__": "builtins",
                    "__loader__": "BytesNode",
                    "file": f"{uuid_name}.bin",
                    "__id__": other,
                },
            },
            "__id__": other + 16,
        },
        "__id__": other + 32,
        "protocol": 2,
        "_skops_version": "0.15.0",
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in (
            (f"{address}.npy", array),
            (f"{uuid_name}.bin", b"\x00planted bytes"),
            ("schema.json", json.dumps(schema, indent=2).encode("utf-8")),
        ):
            archive.writestr(zipfile.ZipInfo(name, date_time=stamp), payload)
    return buffer.getvalue()


class TestTheRewriteNeedsNoSkops:
    """The rewrite is zip and JSON alone, so it is checked on every CI leg,
    including those that do not install the `skops` extra."""

    def test_addresses_uuids_and_clocks_do_not_reach_the_bytes(self):
        first = _skops_shaped(
            2269133876912, 2269122242064, "0f1e", (2026, 10, 4, 9, 25, 38)
        )
        second = _skops_shaped(1407375360, 1407376000, "9abc", (2026, 10, 5, 1, 2, 4))
        assert first != second
        assert deterministic_archive(first) == deterministic_archive(second)

    def test_ids_and_members_are_renamed_together(self):
        rewritten = deterministic_archive(
            _skops_shaped(2269133876912, 2269122242064, "0f1e", (2026, 10, 4, 9, 0, 0))
        )
        with zipfile.ZipFile(io.BytesIO(rewritten)) as archive:
            infos = archive.infolist()
            schema = json.loads(archive.read("schema.json"))
            assert [i.filename for i in infos] == [
                "1.npy",
                "member-1.bin",
                "schema.json",
            ]
            assert {i.date_time for i in infos} == {(1980, 1, 1, 0, 0, 0)}
            assert archive.read("1.npy") == b"\x93NUMPY planted array"
            assert archive.read("member-1.bin") == b"\x00planted bytes"
        # Numbered in the order the schema names them: skops writes an
        # object's `__id__` after its content, so children come first.
        fields = schema["content"]["content"]
        assert fields["coef_"]["file"] == "1.npy" and fields["coef_"]["__id__"] == 1
        # The second reference still names the object the first one does.
        assert fields["coef_again_"]["__id__"] == 1
        assert fields["note_"]["file"] == "member-1.bin"
        assert fields["note_"]["__id__"] == 2
        assert (schema["content"]["__id__"], schema["__id__"]) == (3, 4)
        assert schema["protocol"] == 2

    def test_padding_bytes_are_zeroed_and_fields_kept(self):
        """A forest's node records carry seven bytes no field covers, and
        two dumps of one forest differed there alone."""
        stamp = (2026, 10, 4, 9, 0, 0)
        rewritten = [
            deterministic_archive(
                _skops_shaped(7, 9, "0f1e", stamp, array=_nodes_npy(padding))
            )
            for padding in (0x00, 0xAB)
        ]
        assert rewritten[0] == rewritten[1]
        with zipfile.ZipFile(io.BytesIO(rewritten[1])) as archive:
            payload = archive.read("1.npy")
        planted = _nodes_npy(0xAB)
        header = len(planted) - 3 * _NODE.itemsize
        assert payload[:header] == planted[:header]
        nodes = np.load(io.BytesIO(payload), allow_pickle=False)
        assert nodes.dtype == _NODE
        assert nodes["left"].tolist() == [1, -1, -1]
        assert nodes["threshold"].tolist() == [0.5, -2.0, -2.0]
        assert nodes["flag"].tolist() == [1, 0, 0]
        assert payload[header:] == _nodes_npy(0x00)[header:]


@needs_skops
class TestTheSameModelGivesTheSameBytes:
    """`skops.io.dump` dated every member with the wall clock and named every
    array after its object's memory address, so two runs that fit the same
    model wrote different `model.skops` bytes and the manifest's content hash
    for it never reproduced (the CHANGELOG entry of 2026-10-04)."""

    def test_two_identical_runs_write_the_same_bundle(self, patched_multi_factory):
        first = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_skops_a")
        second = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_skops_b")
        assert first != second
        a, b = load_manifest(first), load_manifest(second)
        assert a.content_hashes["model.skops"] == b.content_hashes["model.skops"]
        assert (_artifacts.run_dir(first) / "model.skops").read_bytes() == (
            _artifacts.run_dir(second) / "model.skops"
        ).read_bytes()
        # Every artifact the two runs wrote hashes alike.
        assert a.content_hashes == b.content_hashes

    def test_separately_fitted_models_give_one_archive(self, tmp_path):
        import skops.io as sio

        for name, model in _fitted()[1].items():
            model = model[0] if isinstance(model, tuple) else model
            again = _fitted()[1][name]
            again = again[0] if isinstance(again, tuple) else again
            raw_a, raw_b = sio.dumps(model), sio.dumps(again)
            # Both models are alive, so their arrays sit at different
            # addresses and skops names them apart.
            assert raw_a != raw_b, name
            assert deterministic_archive(raw_a) == deterministic_archive(raw_b), name
            path = dump_estimator(tmp_path, name, model)
            assert Path(path).read_bytes() == deterministic_archive(raw_b), name

    def test_a_forest_loaded_twice_gives_one_archive(self, tmp_path):
        """Each load of a forest leaves its own bytes in the padding of
        every node record; rewritten, the two archives agree, and the forest
        loaded from one predicts as the original does."""
        import joblib
        import skops.io as sio

        X, models = _fitted()
        forest = models["forest"]
        buffer = io.BytesIO()
        joblib.dump(forest, buffer)
        copies = [joblib.load(io.BytesIO(buffer.getvalue())) for _ in range(2)]
        rewritten = [deterministic_archive(sio.dumps(copy)) for copy in copies]
        assert rewritten[0] == rewritten[1]
        path = tmp_path / "forest.skops"
        path.write_bytes(rewritten[0])
        restored = sio.load(path, trusted=sio.get_untrusted_types(file=path))
        assert np.array_equal(restored.predict(X), forest.predict(X))

    def test_what_loads_is_unchanged(self, tmp_path):
        """The rewritten archive loads to the estimator the original loads
        to: the same predictions, parameters, bytes value and shared
        reference, and the same types to trust."""
        import skops.io as sio

        X, models = _fitted()
        for name, model in models.items():
            model, inputs = model if isinstance(model, tuple) else (model, X)
            original = tmp_path / f"{name}-original.skops"
            sio.dump(model, original)
            rewritten = tmp_path / f"{name}.skops"
            rewritten.write_bytes(deterministic_archive(original.read_bytes()))
            assert untrusted_types(str(rewritten)) == untrusted_types(str(original))
            before, after = load_estimator(str(original)), load_estimator(
                str(rewritten)
            )
            assert np.array_equal(after.predict(inputs), before.predict(inputs)), name
            assert after.get_params().keys() == before.get_params().keys()
        carrying = load_estimator(str(tmp_path / "carrying.skops"))
        assert carrying.note_ == b"\x00planted bytes"
        assert carrying.coef_again_ is carrying.coef_

    def test_an_archive_written_before_still_loads(self, tmp_path):
        """A bundle `skops.io.dump` wrote as it is, timestamps and address
        names included: what every registration before this wrote."""
        import skops.io as sio

        X, models = _fitted()
        path = tmp_path / "earlier.skops"
        sio.dump(models["pipeline"], path)
        with zipfile.ZipFile(path) as archive:
            assert any(
                info.date_time != (1980, 1, 1, 0, 0, 0) for info in archive.infolist()
            )
        restored = load_estimator(str(path))
        assert np.array_equal(restored.predict(X), models["pipeline"].predict(X))

    def test_an_archive_it_does_not_recognise_is_kept_as_written(self):
        """Null case: a zip that is not a skops archive comes back byte for
        byte rather than rewritten into something else."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("other.json", "{}")
        assert deterministic_archive(buffer.getvalue()) == buffer.getvalue()


#: The registered estimators whose fitted state is a LightGBM or XGBoost
#: booster: skops writes them, and loading one would mean trusting that
#: library's own state, so their bundles are refused by name.
_BOOSTER_LIBRARIES = ("lightgbm", "xgboost")


def _is_booster(name):
    return any(library in name for library in _BOOSTER_LIBRARIES)


def _cases():
    """(task, name, calibration) for every registered estimator, and each
    scikit-learn classifier calibrated both ways the spec allows."""
    out = []
    for task, name in sorted(ESTIMATOR_REGISTRY):
        out.append((task, name, None))
        if task == "classification" and not _is_booster(name):
            out.extend([(task, name, "sigmoid"), (task, name, "isotonic")])
    return out


_CASES = _cases()


def _case_id(case):
    task, name, calibration = case
    return f"{task}-{name}" + (f"-{calibration}" if calibration else "")


#: Small fits of the slow estimators: the types a model holds do not depend
#: on how many trees or iterations it has.
_SMALL = {
    "gradient_boosting": {"n_estimators": 10},
    "hist_gradient_boosting": {"max_iter": 10},
    "mlp": {"max_iter": 30},
    "quantile_gradient_boosting": {"n_estimators": 10},
    "random_forest": {"n_estimators": 8},
}


def _fit_registered(case, n=240, seed=3):
    """The estimator built and fitted the way the engine builds and fits
    one, on a small planted problem for its task."""
    from sklearn.calibration import CalibratedClassifierCV

    task, name, calibration = case
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    score = X @ np.array([0.5, -0.3, 0.2, 0.0]) + 0.2 * rng.standard_normal(n)
    y = {
        "regression": score,
        "classification": (score > 0).astype(int),
        "ranking": np.digitize(score, [-0.5, 0.0, 0.5]),
        "survival": np.column_stack(
            [np.exp(-score) + 0.1, (rng.random(n) < 0.7).astype(float)]
        ),
    }[task]
    params = _SMALL.get(name, {})
    model = engine._instantiate(ESTIMATOR_REGISTRY[(task, name)], params, 7, n_jobs=1)
    if calibration:
        model = CalibratedClassifierCV(model, method=calibration, cv=3)
    group = np.full(n // 24, 24) if task == "ranking" else None
    with warnings.catch_warnings():
        # A 30-iteration MLP has not converged, and need not have.
        warnings.simplefilter("ignore")
        engine._fit(model, X, y, None, group=group)
    return model, X


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    """Every case fitted once and written as `dump_estimator` writes it."""
    directory = tmp_path_factory.mktemp("bundles")
    out = {}
    for case in _CASES:
        model, X = _fit_registered(case)
        path = dump_estimator(directory, _case_id(case), model)
        out[case] = (model, X, path)
    return out


@needs_skops
class TestEveryRegisteredEstimatorLoads:
    """`load_estimator` refused every random forest's bundle, and every
    gradient-boosting, histogram-boosting, calibrated and MLP bundle as
    well: the trees, calibrators and optimizer those models hold are
    outside skops' default trusted types (the CHANGELOG entry of
    2026-10-04). It now trusts those five scikit-learn types by name and
    nothing else."""

    @pytest.mark.parametrize(
        "case", [c for c in _CASES if not _is_booster(c[1])], ids=_case_id
    )
    def test_the_bundle_loads_the_same_model(self, bundles, case):
        model, X, path = bundles[case]
        assert path is not None, "the library wrote no bundle"
        restored = load_estimator(path)
        assert type(restored) is type(model)
        assert np.array_equal(restored.predict(X), model.predict(X))
        if hasattr(model, "predict_proba"):
            assert np.array_equal(restored.predict_proba(X), model.predict_proba(X))

    @pytest.mark.parametrize(
        "case", [c for c in _CASES if _is_booster(c[1])], ids=_case_id
    )
    def test_a_booster_is_refused_by_its_type(self, bundles, case):
        _model, _X, path = bundles[case]
        assert path is not None
        with pytest.raises(ValidationError, match=r"(lightgbm|xgboost)\.[\w.]*Booster"):
            load_estimator(path)

    def test_the_trusted_types_are_the_ones_the_estimators_hold(self, bundles):
        """Exactly: every type trusted by name is one some registered
        estimator holds, and every one they hold is trusted."""
        held = set()
        for case, (_model, _X, path) in bundles.items():
            if not _is_booster(case[1]):
                held.update(
                    t for t in untrusted_types(path) if not t.startswith(TRUSTED_PREFIX)
                )
        assert held == set(TRUSTED_TYPES)

    def test_a_scikit_learn_type_the_library_never_writes_is_refused(self, tmp_path):
        """Null case: trusting the Adam optimizer an MLP holds does not
        trust the SGD one, which no registered estimator uses."""
        from sklearn.neural_network import MLPRegressor

        X, models = _fitted()
        model = MLPRegressor(
            hidden_layer_sizes=(4,), solver="sgd", max_iter=5, random_state=0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, models["ridge"].predict(X))
        path = dump_estimator(tmp_path, "mlp_sgd", model)
        with pytest.raises(ValidationError, match="SGDOptimizer"):
            load_estimator(path)

    def test_a_registered_forest_loads_from_its_bundle(self, patched_multi_factory):
        from standard_quant_tools.modeling.specs import (
            EstimatorSpec,
            ModelSpec,
            ValidationSpec,
        )

        model_id = _train_a_model_with_spec(
            _dataset_spec(),
            dataset_id="ds_skops_forest",
            model_spec=ModelSpec(
                task="regression",
                estimator=EstimatorSpec(
                    type="random_forest", params={"n_estimators": 5, "max_depth": 3}
                ),
                validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
                random_seed=1,
            ),
        )
        X = _features(model_id)
        via_skops = load_model(model_id, format="skops")
        assert np.array_equal(via_skops.predict(X), load_model(model_id).predict(X))


@needs_skops
class TestTheBundleOfAHistogramBoostingModel:
    def test_does_not_record_the_fit_s_thread_count(self, tmp_path):
        """The bin mapper's `n_threads` was written into `schema.json`, so
        the same model fitted under two OpenMP limits gave two bundles."""
        import copy

        from sklearn.ensemble import HistGradientBoostingRegressor

        X, models = _fitted()
        model = HistGradientBoostingRegressor(max_iter=10, random_state=0).fit(
            X, models["ridge"].predict(X)
        )
        twins = [copy.deepcopy(model) for _ in range(2)]
        twins[0]._bin_mapper.n_threads = 2
        twins[1]._bin_mapper.n_threads = 9
        paths = [dump_estimator(tmp_path, f"hgb{i}", m) for i, m in enumerate(twins)]
        assert Path(paths[0]).read_bytes() == Path(paths[1]).read_bytes()
        assert [m._bin_mapper.n_threads for m in twins] == [2, 9]
        restored = load_estimator(paths[0])
        assert restored._bin_mapper.n_threads is None
        assert np.array_equal(restored.predict(X), model.predict(X))
