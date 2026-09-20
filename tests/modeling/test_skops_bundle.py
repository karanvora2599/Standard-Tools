"""
The skops bundle: the registered estimator, loadable without pickle.

joblib is pickle, and pickle executes code from the file. The bundle is
the same estimator written as declared state, and the loader constructs
only the types it was told to trust. Planted: a tampered joblib is
refused while the bundle still loads, a tampered bundle is refused before
it is read, and a bundle that names a type from outside this package is
refused by that type's name.
"""

import numpy as np
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.capabilities import modeling_capabilities
from standard_quant_tools.modeling.estimators.survival import CoxPHRegressor
from standard_quant_tools.modeling.registry.model_registry import (
    load_manifest,
    load_model,
    load_monitoring_reference,
)
from standard_quant_tools.modeling.registry.serialization import (
    FORMAT_ENV,
    dump_estimator,
    load_estimator,
    skops_available,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec
from .test_survival import _planted

pytestmark = pytest.mark.skipif(not skops_available(), reason="skops is not installed")


class _NotOurs:
    """A type this registry never writes."""

    def __init__(self):
        self.value = 1


def _features(model_id):
    manifest = load_manifest(model_id)
    _profile, reference, _predictions = load_monitoring_reference(model_id)
    return reference[manifest.feature_ids].to_numpy(dtype=float)


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
