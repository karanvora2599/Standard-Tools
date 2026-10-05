"""
The skops bundle: the registered estimator, loadable without pickle.

joblib is pickle, and pickle executes code from the file. The bundle is
the same estimator written as declared state, and the loader constructs
only the types it was told to trust. Planted: a tampered joblib is
refused while the bundle still loads, a tampered bundle is refused before
it is read, and a bundle that names a type from outside this package is
refused by that type's name -- and is not written by the registry. A
fitted model and that model loaded from either of its files write one
bundle.
"""

import io
import json
import re
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
from standard_quant_tools.modeling.registry.package import verify_model_package
from standard_quant_tools.modeling.registry.serialization import (
    FORMAT_ENV,
    TRUSTED_PREFIX,
    TRUSTED_TYPES,
    TRUSTED_TYPES_SKOPS,
    deterministic_archive,
    dump_estimator,
    load_estimator,
    save_joblib,
    skops_available,
    state_hash,
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

    def test_a_bundle_skops_cannot_read_back_is_not_written(
        self, tmp_path, monkeypatch, caplog
    ):
        """The types a bundle holds are read before it is written (the
        CHANGELOG entry of 2026-10-04); when skops cannot read them, the
        model keeps joblib alone rather than failing its registration."""
        import skops.io as sio

        def _cannot(*_args, **_kwargs):
            raise ValueError("planted: a node skops cannot audit")

        monkeypatch.setattr(sio, "get_untrusted_types", _cannot)
        with caplog.at_level("WARNING"):
            assert dump_estimator(tmp_path, "ridge", _fitted()[1]["ridge"]) is None
        assert "could not read back" in caplog.text
        assert not (tmp_path / "ridge.skops").exists()

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
    stored under its object's address, a bytes value under a UUID,
    `schema.json` last, every member dated `stamp`; and a second reference
    to the array as a reference node (`CachedNode`), which skops' loader
    reads and the rewrite numbers by the id it names."""
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


def _array_npy(values) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values, dtype=float), allow_pickle=False)
    return buffer.getvalue()


def _json_node(content: str, node_id: int) -> dict:
    return {
        "__class__": "str",
        "__module__": "builtins",
        "__loader__": "JsonNode",
        "content": content,
        "is_json": True,
        "__id__": node_id,
    }


def _array_node(node_id: int) -> dict:
    return {
        "__class__": "ndarray",
        "__module__": "numpy",
        "__loader__": "NdArrayNode",
        "type": "numpy",
        "file": f"{node_id}.npy",
        "__id__": node_id,
    }


def _shared_or_not(shared: bool, tol_value: str = "0.5") -> bytes:
    """One model as skops 0.15 writes it, with or without the objects a fit
    shares: `coef_` and `coef_again_` one array (one `__id__`, one member)
    or two equal ones, and `alpha` and `tol` one float object or two equal
    ones. skops writes every reference in full under the object's
    address."""
    first, second = 2269133876912, (2269133876912 if shared else 2269133877104)
    alpha, tol = 1407375360, (1407375360 if shared else 1407376000)
    schema = {
        "__class__": "Ridge",
        "__module__": "sklearn.linear_model._ridge",
        "__loader__": "ObjectNode",
        "content": {
            "__class__": "dict",
            "__module__": "builtins",
            "__loader__": "DictNode",
            "content": {
                "coef_": _array_node(first),
                "coef_again_": _array_node(second),
                "alpha": _json_node("0.5", alpha),
                "tol": _json_node(tol_value, tol),
            },
            "__id__": 9001,
        },
        "__id__": 9002,
        "protocol": 2,
        "_skops_version": "0.15.0",
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for address in dict.fromkeys([first, second]):
            archive.writestr(f"{address}.npy", _array_npy([1.0, -2.0]))
        archive.writestr("schema.json", json.dumps(schema, indent=2))
    return buffer.getvalue()


class TestTheRewriteNeedsNoSkops:
    """The rewrite is zip and JSON alone, so it is checked on every CI leg,
    including those that do not install the `skops` extra."""

    def test_which_objects_a_model_shared_does_not_reach_the_bytes(self):
        """A fit shares objects a load holds as copies, and skops records
        the sharing in its ids (the CHANGELOG entry of 2026-10-04). Both
        archives rewrite to one: the arrays one id and one member each,
        the equal floats one id."""
        shared, separate = _shared_or_not(True), _shared_or_not(False)
        assert shared != separate
        rewritten = deterministic_archive(shared)
        assert rewritten == deterministic_archive(separate)
        assert deterministic_archive(rewritten) == rewritten
        with zipfile.ZipFile(io.BytesIO(rewritten)) as archive:
            names = [info.filename for info in archive.infolist()]
            schema = json.loads(archive.read("schema.json"))
            assert names == ["1.npy", "2.npy", "schema.json"]
            assert archive.read("1.npy") == archive.read("2.npy")
        fields = schema["content"]["content"]
        assert (fields["coef_"]["__id__"], fields["coef_"]["file"]) == (1, "1.npy")
        assert (fields["coef_again_"]["__id__"], fields["coef_again_"]["file"]) == (
            2,
            "2.npy",
        )
        assert fields["alpha"]["__id__"] == fields["tol"]["__id__"] == 3
        assert (schema["content"]["__id__"], schema["__id__"]) == (4, 5)

    def test_values_that_differ_keep_their_own_ids(self):
        """Null case: two floats that are not equal are two values."""
        rewritten = deterministic_archive(_shared_or_not(False, tol_value="0.25"))
        with zipfile.ZipFile(io.BytesIO(rewritten)) as archive:
            fields = json.loads(archive.read("schema.json"))["content"]["content"]
        assert (fields["alpha"]["__id__"], fields["tol"]["__id__"]) == (3, 4)

    def test_the_extra_requires_the_release_the_trusted_types_were_read_off(self):
        """`TRUSTED_TYPES` was read off skops 0.15, and the `skops` extra
        allowed 0.10, whose defaults can name types the loader refuses
        (the CHANGELOG entry of 2026-10-04)."""
        pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        [floor] = re.findall(r'^\s*"skops>=([0-9.]+)"', pyproject, flags=re.MULTILINE)
        assert floor == TRUSTED_TYPES_SKOPS

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
        to: the same predictions, parameters and bytes value, and the same
        types to trust.

        Which objects are one object is what changes (the CHANGELOG entry
        of 2026-10-04): an array the fitted model held under two names
        loads as two equal arrays, as it does from the model's joblib,
        because the bundle records what the model holds, not which of its
        objects a fit happened to share."""
        import joblib
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
        assert np.array_equal(carrying.coef_again_, carrying.coef_)
        assert carrying.coef_again_ is not carrying.coef_
        # The joblib of the same model loads it the same way.
        buffer = io.BytesIO()
        joblib.dump(models["carrying"], buffer)
        via_joblib = joblib.load(io.BytesIO(buffer.getvalue()))
        assert via_joblib.coef_again_ is not via_joblib.coef_

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
#: library's own state, so their bundles are refused by name -- and,
#: since the CHANGELOG entry of 2026-10-04, not written.
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
    def test_a_booster_is_not_written_and_is_refused_by_its_type(
        self, bundles, case, tmp_path, caplog
    ):
        """Registration wrote a bundle the loader then refused, and the
        manifest listed it as a format the model has (the CHANGELOG entry
        of 2026-10-04). No bundle is written now, and the log names the
        type; a bundle skops writes for it is still refused by that name."""
        import skops.io as sio

        model, _X, path = bundles[case]
        assert path is None
        with caplog.at_level("WARNING"):
            assert dump_estimator(tmp_path, "again", model) is None
        assert re.search(r"(lightgbm|xgboost)\.[\w.]*Booster", caplog.text)
        assert "registered with joblib only" in caplog.text
        assert not (tmp_path / "again.skops").exists()
        assert state_hash(model) is None
        written = tmp_path / "written.skops"
        written.write_bytes(deterministic_archive(sio.dumps(model)))
        with pytest.raises(ValidationError, match=r"(lightgbm|xgboost)\.[\w.]*Booster"):
            load_estimator(str(written))

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
        trust the SGD one, which no registered estimator uses: its bundle
        is refused by name, and the registry does not write one."""
        import skops.io as sio
        from sklearn.neural_network import MLPRegressor

        X, models = _fitted()
        model = MLPRegressor(
            hidden_layer_sizes=(4,), solver="sgd", max_iter=5, random_state=0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, models["ridge"].predict(X))
        assert dump_estimator(tmp_path, "mlp_sgd", model) is None
        path = tmp_path / "mlp_sgd.skops"
        path.write_bytes(deterministic_archive(sio.dumps(model)))
        with pytest.raises(ValidationError, match="SGDOptimizer"):
            load_estimator(str(path))

    def test_a_refusal_under_an_older_skops_names_the_release(
        self, tmp_path, monkeypatch
    ):
        """`TRUSTED_TYPES` was read off skops 0.15; under an older skops,
        whose defaults can name other types, the refusal says so (the
        CHANGELOG entry of 2026-10-04). Under this release it does not."""
        import skops
        import skops.io as sio

        path = tmp_path / "foreign.skops"
        sio.dump(_NotOurs(), path)
        with pytest.raises(ValidationError) as current:
            load_estimator(str(path))
        assert "derived on skops" not in str(current.value)
        monkeypatch.setattr(skops, "__version__", "0.12.1")
        with pytest.raises(ValidationError) as older:
            load_estimator(str(path))
        assert "_NotOurs" in str(older.value)
        assert (
            f"derived on skops {TRUSTED_TYPES_SKOPS}; this process has skops 0.12.1"
            in str(older.value)
        )

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


def _distinct_ids(raw: bytes) -> int:
    """How many distinct `__id__` values a skops archive's schema holds."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        text = archive.read("schema.json").decode("utf-8")
    return len(set(re.findall(r'"__id__": (\d+)', text)))


@needs_skops
class TestOneBundleWhereverTheModelCameFrom:
    """A fitted model, the same model loaded from its joblib and the same
    model loaded from its bundle wrote three different bundles: skops' ids
    record which objects the model in memory shares, and a fit, a joblib
    load and a skops load share different ones -- a float or a string a
    fit passes between attributes, an array joblib loads once per
    reference, the `inf` a skops load reads as one object (the CHANGELOG
    entry of 2026-10-04). The bundle is the model's state, so all three
    write one, for every registered estimator that has a bundle."""

    @pytest.mark.parametrize(
        "case", [c for c in _CASES if not _is_booster(c[1])], ids=_case_id
    )
    def test_a_fit_and_its_loads_write_one_bundle(self, bundles, case, tmp_path):
        import joblib

        model, X, path = bundles[case]
        fitted = Path(path).read_bytes()
        loads = {
            "joblib": joblib.load(save_joblib(tmp_path, "model", model)),
            "skops": load_estimator(path),
        }
        for name, loaded in loads.items():
            assert Path(dump_estimator(tmp_path, name, loaded)).read_bytes() == fitted
            assert np.array_equal(loaded.predict(X), model.predict(X)), name
        recorded = _artifacts.hash_file(Path(path))
        assert state_hash(model) == recorded
        assert {state_hash(loaded) for loaded in loads.values()} == {recorded}

    def test_the_sharing_the_rewrite_leaves_out_differs(self, bundles):
        """Null case beside it: a calibrated classifier's fit hands each
        fold's calibrated model the array its own `classes_` is, a joblib
        load holds a copy in each, and the two archives skops writes group
        their ids differently."""
        import joblib
        import skops.io as sio

        model = bundles[("classification", "logistic", "sigmoid")][0]
        assert all(c.classes is model.classes_ for c in model.calibrated_classifiers_)
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        loaded = joblib.load(io.BytesIO(buffer.getvalue()))
        assert not any(
            c.classes is loaded.classes_ for c in loaded.calibrated_classifiers_
        )
        assert _distinct_ids(sio.dumps(model)) != _distinct_ids(sio.dumps(loaded))


def _lightgbm_spec(params=None):
    from standard_quant_tools.modeling.specs import (
        EstimatorSpec,
        ModelSpec,
        ValidationSpec,
    )

    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="lightgbm", params=params or {"n_estimators": 20}),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=1,
    )


@needs_skops
class TestARegisteredBooster:
    """A LightGBM or XGBoost model was registered with `formats: ["joblib",
    "skops"]` and a bundle the loader refuses by its booster type (the
    CHANGELOG entry of 2026-10-04)."""

    def test_is_registered_with_joblib_alone(self, patched_multi_factory, caplog):
        pytest.importorskip("lightgbm")
        with caplog.at_level("WARNING"):
            model_id = _train_a_model_with_spec(
                _dataset_spec(),
                dataset_id="ds_skops_booster",
                model_spec=_lightgbm_spec(),
            )
        manifest = load_manifest(model_id)
        assert manifest.formats == ["joblib"]
        assert "model.skops" not in manifest.content_hashes
        assert not (_artifacts.run_dir(model_id) / "model.skops").exists()
        assert "lightgbm.basic.Booster" in caplog.text
        with pytest.raises(ValidationError, match="no skops bundle"):
            load_model(model_id, format="skops")
        assert load_model(model_id).predict(_features(model_id)).shape[0] > 0

    def test_a_manifest_listing_its_bundle_still_loads_and_verifies(
        self, patched_multi_factory
    ):
        """What a registration before wrote: the bundle beside the joblib,
        hashed and listed. The joblib still loads and verifies, the package
        verifies, and the bundle is refused by its type, as it always was."""
        import skops.io as sio

        pytest.importorskip("lightgbm")
        model_id = _train_a_model_with_spec(
            _dataset_spec(),
            dataset_id="ds_skops_booster_old",
            model_spec=_lightgbm_spec(),
        )
        directory = _artifacts.run_dir(model_id)
        model = load_model(model_id)
        bundle = directory / "model.skops"
        bundle.write_bytes(deterministic_archive(sio.dumps(model)))
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["formats"] = ["joblib", "skops"]
        manifest["content_hashes"]["model.skops"] = _artifacts.hash_file(bundle)
        _artifacts.save_json(directory, "manifest", manifest)
        X = _features(model_id)
        assert np.array_equal(load_model(model_id).predict(X), model.predict(X))
        report = verify_model_package(model_id)
        assert "model.skops" in report.verified
        assert not report.mismatched and not report.missing
        with pytest.raises(ValidationError, match=r"lightgbm\.basic\.Booster"):
            load_model(model_id, format="skops")


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
