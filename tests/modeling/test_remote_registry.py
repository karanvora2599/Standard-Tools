"""
A registry that reaches another machine: mirror on register, pull on
demand, verified on the way in.

The store is fsspec's in-memory filesystem, which is the closest thing to
a bucket that runs in-tree. Planted: a file changed on the remote is
refused before anything is registered locally, an unsigned package is
refused when a signature is required and nothing is registered, and a
pull that succeeds loads a model whose predictions are the original's.
"""

from uuid import uuid4

import numpy as np
import pytest

from standard_quant_tools.artifact_store import fsspec_available, store_from_url
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import ListModelsInput
from standard_quant_tools.modeling.agent.tools import list_models
from standard_quant_tools.modeling.registry.lifecycle import current_stage, promote
from standard_quant_tools.modeling.registry.mirror import MIRROR_URL_ENV
from standard_quant_tools.modeling.registry.model_registry import (
    load_manifest,
    load_model,
    load_monitoring_reference,
)
from standard_quant_tools.modeling.registry.package import (
    list_remote_models,
    mirror_model_package,
    pull_model_package,
)
from standard_quant_tools.modeling.registry.signing import (
    sign_manifest,
    signing_available,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec

pytestmark = pytest.mark.skipif(
    not fsspec_available(), reason="fsspec is not installed"
)


def _memory_url() -> str:
    return f"memory://sqt-registry/{uuid4().hex}"


def _filename(entry: str) -> str:
    return entry if "." in entry else f"{entry}.parquet"


def _features(model_id):
    manifest = load_manifest(model_id)
    _profile, reference, _predictions = load_monitoring_reference(model_id)
    return reference[manifest.feature_ids].to_numpy(dtype=float)


class TestTheMirror:
    def test_registration_and_promotion_reach_the_mirror(
        self, patched_multi_factory, monkeypatch
    ):
        url = _memory_url()
        monkeypatch.setenv(MIRROR_URL_ENV, url)
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_mirror_env")
        store = store_from_url(url)
        assert list_remote_models(store) == [model_id]
        manifest = load_manifest(model_id)
        for entry, digest in manifest.content_hashes.items():
            assert store.hash(f"{model_id}/{_filename(entry)}") == digest
        promote(model_id, "validated", "the mirror must carry the stage too")
        local_log = (_artifacts.run_dir(model_id) / "promotions.jsonl").read_bytes()
        assert store.get(f"{model_id}/promotions.jsonl") == local_log


class TestThePull:
    def test_a_pull_registers_a_verified_copy_elsewhere(
        self, patched_multi_factory, monkeypatch, tmp_path
    ):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_pull")
        promote(model_id, "validated", "promoted before it travelled")
        X = _features(model_id)
        expected = load_model(model_id).predict(X)
        store = store_from_url(_memory_url())
        mirror_model_package(model_id, store)

        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "elsewhere"))
        with pytest.raises(ValidationError, match="no registered model"):
            load_manifest(model_id)
        report = pull_model_package(model_id, store)
        assert report.ok and report.model_id == model_id
        assert load_manifest(model_id).model_id == model_id
        assert np.allclose(load_model(model_id).predict(X), expected)
        assert current_stage(model_id) == "validated"
        assert model_id in {m.model_id for m in list_models(ListModelsInput()).models}
        with pytest.raises(ValidationError, match="already registered"):
            pull_model_package(model_id, store)
        assert pull_model_package(model_id, store, overwrite=True).ok

    def test_a_changed_remote_file_registers_nothing(
        self, patched_multi_factory, monkeypatch, tmp_path
    ):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_pull_bad")
        store = store_from_url(_memory_url())
        mirror_model_package(model_id, store)
        store.put(f"{model_id}/model.joblib", b"not the model")

        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "elsewhere"))
        with pytest.raises(ValidationError, match="model.joblib .* does not match"):
            pull_model_package(model_id, store)
        assert not (tmp_path / "elsewhere" / model_id / "manifest.json").exists()
        with pytest.raises(ValidationError, match="no registered model"):
            load_manifest(model_id)

    def test_a_missing_remote_file_and_a_missing_model_are_named(
        self, patched_multi_factory, monkeypatch, tmp_path
    ):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_pull_gap")
        store = store_from_url(_memory_url())
        with pytest.raises(ValidationError, match="does not exist"):
            pull_model_package(model_id, store)
        mirror_model_package(model_id, store)
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "elsewhere"))
        # A store whose listing lacks a hashed file: refused before any write.
        gapped = store_from_url(_memory_url())
        for key in store.list(model_id):
            if not key.endswith("preprocessing_stats.json"):
                gapped.put(key, store.get(key))
        with pytest.raises(
            ValidationError, match="lacks \\['preprocessing_stats.json'\\]"
        ):
            pull_model_package(model_id, gapped)
        assert not (tmp_path / "elsewhere" / model_id).exists()

    @pytest.mark.skipif(not signing_available(), reason="cryptography is not installed")
    def test_a_required_signature_is_checked_on_the_store_s_bytes(
        self, patched_multi_factory, monkeypatch, tmp_path
    ):
        from standard_quant_tools.audit.signing import generate_keypair

        private, public = generate_keypair()
        _other_private, other_public = generate_keypair()
        key_path = tmp_path / "signing.key"
        key_path.write_bytes(private)

        unsigned = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_pull_unsigned"
        )
        unsigned_store = store_from_url(_memory_url())
        mirror_model_package(unsigned, unsigned_store)
        signed = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_pull_signed")
        sign_manifest(signed, key_path=key_path)
        signed_store = store_from_url(_memory_url())
        mirror_model_package(signed, signed_store)

        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "elsewhere"))
        with pytest.raises(ValidationError, match="not signed"):
            pull_model_package(unsigned, unsigned_store, require_signature=True)
        assert not (tmp_path / "elsewhere" / unsigned).exists()
        with pytest.raises(ValidationError, match="not the pinned"):
            pull_model_package(
                signed, signed_store, require_signature=True, public_key=other_public
            )
        assert not (tmp_path / "elsewhere" / signed / "manifest.json").exists()
        report = pull_model_package(
            signed, signed_store, require_signature=True, public_key=public
        )
        assert report.ok and report.signature["key_pinned"] is True
