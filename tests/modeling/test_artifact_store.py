"""
The artifact store and the package operations written against it.

The local store is the one the runtime uses, so it is tested for the two
properties every integrity check rests on: a key cannot leave the root,
and a put is whole or absent. The fsspec store is tested against the
in-memory filesystem fsspec ships, which is the closest thing to a bucket
that runs in-tree. The package operations plant a swapped binary and a
missing file and expect both named, and mirror a package to both stores
expecting the registered digests to be what lands.
"""

from pathlib import Path
from uuid import uuid4

import pytest

from standard_quant_tools.artifact_store import (
    ArtifactStore,
    FsspecArtifactStore,
    LocalArtifactStore,
    fsspec_available,
    hash_bytes,
    store_from_url,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import InspectModelInput
from standard_quant_tools.modeling.agent.tools import inspect_model
from standard_quant_tools.modeling.registry.lifecycle import promote
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.registry.package import (
    mirror_model_package,
    verify_model_package,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec


def _filename(entry: str) -> str:
    return entry if "." in entry else f"{entry}.parquet"


class TestLocalStore:
    def test_round_trip_listing_and_hash(self, tmp_path):
        store = LocalArtifactStore(tmp_path / "store")
        uri = store.put("run_a/one.json", b'{"a": 1}')
        assert Path(uri) == (tmp_path / "store" / "run_a" / "one.json").resolve()
        assert store.get("run_a/one.json") == b'{"a": 1}'
        assert store.exists("run_a/one.json") and not store.exists("run_a/two.json")
        store.put("run_a/two.parquet", b"pq")
        store.put("run_b/x.joblib", b"jb")
        assert store.list() == ["run_a/one.json", "run_a/two.parquet", "run_b/x.joblib"]
        assert store.list("run_a") == ["run_a/one.json", "run_a/two.parquet"]
        assert store.list("run_z") == []
        assert store.hash("run_a/one.json") == hash_bytes(b'{"a": 1}')
        assert store.hash("run_a/one.json") == _artifacts.hash_file(Path(uri))
        assert isinstance(store, ArtifactStore)
        with pytest.raises(ValidationError, match="not found"):
            store.get("run_a/none.json")
        with pytest.raises(ValidationError, match="not found"):
            store.hash("run_a/none.json")

    def test_a_key_cannot_leave_the_root(self, tmp_path):
        root = tmp_path / "root"
        store = LocalArtifactStore(root)
        for bad in (
            "../x.json",
            "run/../../x.json",
            "/abs/x.json",
            "run\\x.json",
            "run/.hidden",
            "run/a/b.json",
            "",
            "run/",
            "run/..",
        ):
            with pytest.raises(ValidationError):
                store.put(bad, b"x")
        assert not (tmp_path / "x.json").exists()
        assert not root.exists() or list(root.iterdir()) == []

    def test_a_put_is_whole_and_leaves_nothing_behind(self, tmp_path):
        store = LocalArtifactStore(tmp_path)
        store.put("r/f.bin", b"first")
        store.put("r/f.bin", b"second and longer")
        assert store.get("r/f.bin") == b"second and longer"
        assert [p.name for p in (tmp_path / "r").iterdir()] == ["f.bin"]

    def test_the_default_root_follows_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        store = LocalArtifactStore()
        store.put("r/a.txt", b"a")
        assert (tmp_path / "runs" / "r" / "a.txt").read_bytes() == b"a"
        assert isinstance(store_from_url(str(tmp_path)), LocalArtifactStore)
        assert isinstance(
            store_from_url(f"file://{tmp_path.as_posix()}"), LocalArtifactStore
        )


@pytest.mark.skipif(not fsspec_available(), reason="fsspec is not installed")
class TestFsspecStore:
    def test_memory_store_round_trip(self):
        store = FsspecArtifactStore(f"memory://sqt-tests/{uuid4().hex}")
        assert isinstance(store, ArtifactStore)
        uri = store.put("run_a/one.json", b"abc")
        assert uri.startswith("memory://")
        assert store.get("run_a/one.json") == b"abc"
        assert store.exists("run_a/one.json") and not store.exists("run_a/two.json")
        store.put("run_a/two.parquet", b"pq")
        assert store.list() == ["run_a/one.json", "run_a/two.parquet"]
        assert store.list("run_a") == ["run_a/one.json", "run_a/two.parquet"]
        assert store.list("nope") == []
        assert store.hash("run_a/one.json") == hash_bytes(b"abc")
        with pytest.raises(ValidationError, match="not found"):
            store.get("run_a/none.json")
        with pytest.raises(ValidationError):
            store.put("../escape.json", b"x")

    def test_store_from_url_picks_the_fsspec_store(self):
        assert isinstance(
            store_from_url("memory://sqt-tests/pick"), FsspecArtifactStore
        )


class TestPackage:
    def test_a_registered_package_verifies_whole(self, patched_multi_factory):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_pkg")
        report = verify_model_package(model_id)
        assert report.ok and not report.missing and not report.mismatched
        assert {
            "model.joblib",
            "model_spec.json",
            "preprocessing_stats.json",
            "dataset_spec.json",
            "oos_predictions.parquet",
            "feature_reference.parquet",
            "prediction_reference.parquet",
            "feature_profile.json",
        } <= set(report.verified)
        assert report.signature is None and report.signature_error is None
        assert "manifest.json" not in report.unhashed
        assert "manifest.json" not in report.verified
        lineage = inspect_model(
            InspectModelInput(model_id=model_id, view="lineage")
        ).data
        assert lineage["package"]["ok"] is True
        assert lineage["package"]["verified"] == report.verified

    def test_a_swapped_binary_and_a_missing_file_are_named(
        self, patched_multi_factory, tmp_path
    ):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_tamper")
        directory = _artifacts.run_dir(model_id)
        with open(directory / "model.joblib", "ab") as handle:
            handle.write(b"\x00")
        (directory / "preprocessing_stats.json").unlink()
        report = verify_model_package(model_id)
        assert report.mismatched == ["model.joblib"]
        assert report.missing == ["preprocessing_stats.json"]
        assert not report.ok
        assert report.to_dict()["ok"] is False
        with pytest.raises(ValidationError, match="does not verify"):
            mirror_model_package(model_id, LocalArtifactStore(tmp_path / "mirror"))
        assert not (tmp_path / "mirror").exists()
        # Requiring a signature of an unsigned package is a finding, not a crash.
        required = verify_model_package(model_id, require_signature=True)
        assert required.signature_error is not None and not required.ok

    def test_a_mirror_lands_with_the_registered_hashes(
        self, patched_multi_factory, tmp_path
    ):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_mirror")
        promote(model_id, "validated", "so the mirror carries a promotion log")
        target = LocalArtifactStore(tmp_path / "mirror")
        uris = mirror_model_package(model_id, target)
        source_files = {
            key.split("/", 1)[1] for key in LocalArtifactStore().list(model_id)
        }
        assert set(uris) == source_files
        assert {"promotions.jsonl", "manifest.json"} <= set(uris)
        manifest = load_manifest(model_id)
        for entry, digest in manifest.content_hashes.items():
            assert target.hash(f"{model_id}/{_filename(entry)}") == digest
        assert (
            target.get(f"{model_id}/manifest.json")
            == (_artifacts.run_dir(model_id) / "manifest.json").read_bytes()
        )
        with pytest.raises(ValidationError):
            mirror_model_package(model_id, target, prefix="../elsewhere")

    @pytest.mark.skipif(not fsspec_available(), reason="fsspec is not installed")
    def test_a_mirror_to_an_object_store_is_verified_through_it(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_bucket")
        target = FsspecArtifactStore(f"memory://sqt-mirror/{uuid4().hex}")
        uris = mirror_model_package(model_id, target, prefix="mirror_a")
        assert all(uri.startswith("memory://") for uri in uris.values())
        manifest = load_manifest(model_id)
        assert (
            target.hash("mirror_a/model.joblib")
            == manifest.content_hashes["model.joblib"]
        )
        assert target.list("mirror_a") == sorted(f"mirror_a/{name}" for name in uris)
