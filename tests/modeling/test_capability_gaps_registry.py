"""
A model package that travels, and arrives whole.

WHAT THESE PIN.

  1. A manifest records its Parquet artifacts by FILENAME inside the
     model's own directory. Recorded as absolute paths of the registering
     machine, they were unreadable everywhere else: in another runs root
     on the same machine the containment check refused the old root's
     file, and on another machine monitoring reported that the model "was
     registered before monitoring references were kept" and told the
     operator to retrain -- while `feature_reference.parquet` sat beside
     the manifest it was named in. A pulled model can now be scored,
     monitored, simulated as a portfolio and backtested in the root it
     was pulled into.
  2. The resolver still reads a manifest written the old way, as long as
     the path it names is inside this runs root; one that is not is
     refused by name rather than followed.
  3. A reference the manifest names and the directory does not have is a
     BROKEN package, not an old one, and says so. Only a manifest with no
     reference at all is the model that predates them.
  4. The mirror is reachable from the tool surface: a store can be
     listed, a package pulled and verified, and every refusal the pull
     already produced -- unsigned when required, already registered,
     the wrong pinned key, a nested key no flat store can hold -- comes
     back as a refusal naming the remedy, with nothing registered.
"""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.artifact_store import fsspec_available, store_from_url
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import (
    MonitorModelInput,
    ScoreModelInput,
)
from standard_quant_tools.modeling.agent.remote_models import (
    ListRemoteModelsInput,
    PullModelPackageInput,
)
from standard_quant_tools.modeling.agent.remote_tools import (
    list_remote_models,
    pull_model_package,
)
from standard_quant_tools.modeling.agent.tools import monitor_model, score_model
from standard_quant_tools.modeling.bridge import oos_predictions_to_signal_panel
from standard_quant_tools.modeling.portfolio_eval import evaluate_model_portfolio
from standard_quant_tools.modeling.registry.lifecycle import promote
from standard_quant_tools.modeling.registry.mirror import MIRROR_URL_ENV
from standard_quant_tools.modeling.registry.model_registry import (
    load_manifest,
    load_monitoring_reference,
    resolve_model_artifact,
)
from standard_quant_tools.modeling.registry.package import (
    mirror_model_package,
)
from standard_quant_tools.modeling.registry.package import (
    pull_model_package as pull_package,
)
from standard_quant_tools.modeling.registry.signing import (
    sign_manifest,
    signing_available,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec

pytestmark = pytest.mark.skipif(
    not fsspec_available(), reason="fsspec is not installed"
)

UNIVERSE = ["AAA", "BBB", "CCC"]


def _memory_url() -> str:
    """A store per test: the in-memory filesystem is process-global, so a
    shared prefix would let one test's package show up in another's
    listing."""
    return f"memory://sqt-packages/{uuid4().hex}"


def _edit_manifest(model_id: str, edit) -> None:
    """Rewrite a registered manifest the way an older version of this
    package would have written it. The manifest is the root of the
    content hashes and cannot contain its own, so editing it leaves every
    other digest intact -- which is exactly the situation a model
    registered before the filenames moved is in."""
    directory = _artifacts.run_dir(model_id)
    payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    edit(payload)
    _artifacts.save_json(directory, "manifest", payload)


class TestAPulledPackageReadsItsOwnArtifacts:
    def test_the_monitoring_references_survive_a_change_of_runs_root(
        self, patched_multi_factory, monkeypatch, tmp_path
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_travelling_references"
        )
        manifest = load_manifest(model_id)
        # The filenames the package carries, not the paths of the machine
        # that wrote them.
        assert manifest.monitoring["feature_reference_uri"] == (
            "feature_reference.parquet"
        )
        assert manifest.monitoring["prediction_reference_uri"] == (
            "prediction_reference.parquet"
        )
        assert manifest.oos_predictions_uri == "oos_predictions.parquet"
        profile, features, predictions = load_monitoring_reference(model_id)
        store = store_from_url(_memory_url())
        mirror_model_package(model_id, store)

        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "another_root"))
        assert pull_package(model_id, store).ok
        pulled_profile, pulled_features, pulled_predictions = load_monitoring_reference(
            model_id
        )
        assert pulled_profile == profile
        pd.testing.assert_frame_equal(pulled_features, features)
        pd.testing.assert_frame_equal(pulled_predictions, predictions)

    def test_a_pulled_model_scores_and_monitors_where_it_landed(
        self, patched_multi_factory, monkeypatch, tmp_path
    ):
        """The gap this closes, end to end: the pulled model was the one
        whose monitoring reported that it had no references and advised a
        retrain."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_monitored_after_pull"
        )
        store = store_from_url(_memory_url())
        mirror_model_package(model_id, store)

        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "another_root"))
        assert pull_package(model_id, store).ok
        scored = score_model(
            ScoreModelInput(model_id=model_id, as_of="2023-12-29", universe=UNIVERSE)
        )
        report = monitor_model(
            MonitorModelInput(model_id=model_id, predictions_uri=scored.predictions_uri)
        )
        assert [row.feature for row in report.feature_drift] == (
            load_manifest(model_id).feature_ids
        )
        for row in report.feature_drift:
            assert np.isfinite(row.psi) and row.psi >= 0.0
            assert row.status in {"stable", "moderate", "severe"}
        assert report.prediction_drift["n_reference"] > 0
        assert report.overall_status in {"stable", "moderate", "severe"}

    def test_the_portfolio_and_the_bridge_read_a_pulled_model_too(
        self, patched_multi_factory, monkeypatch, tmp_path
    ):
        """Both readers of `oos_predictions_uri` that produce numbers: the
        simulator and the verified branch of the backtest bridge."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_simulated_after_pull"
        )
        here = evaluate_model_portfolio(model_id)
        here_panel = oos_predictions_to_signal_panel(model_id=model_id)
        store = store_from_url(_memory_url())
        mirror_model_package(model_id, store)

        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "another_root"))
        assert pull_package(model_id, store).ok
        there = evaluate_model_portfolio(model_id)
        # The same predictions produce the same book, and the provenance
        # names the file that was actually read -- this root's.
        assert there["metrics"]["sharpe_ratio"] == pytest.approx(
            here["metrics"]["sharpe_ratio"], rel=1e-12, nan_ok=True
        )
        assert there["provenance"]["oos_predictions_hash"] == (
            load_manifest(model_id).content_hashes["oos_predictions"]
        )
        assert there["provenance"]["oos_predictions_uri"].startswith(
            str(tmp_path / "another_root")
        )
        assert oos_predictions_to_signal_panel(model_id=model_id) == here_panel


class TestTheResolverReadsEveryFormAManifestEverHad:
    def test_a_legacy_absolute_path_in_this_root_still_resolves(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_legacy_absolute_uri"
        )
        directory = _artifacts.run_dir(model_id)
        expected = load_monitoring_reference(model_id)[1]
        _edit_manifest(
            model_id,
            lambda payload: payload["monitoring"].update(
                feature_reference_uri=str(directory / "feature_reference.parquet")
            ),
        )
        pd.testing.assert_frame_equal(load_monitoring_reference(model_id)[1], expected)

        # The same path, for a file that is no longer in the model's own
        # directory but is still inside this runs root: the fallback the
        # containment check guards.
        elsewhere = _artifacts.run_dir("mdl_archived_reference_copy")
        elsewhere.mkdir(parents=True, exist_ok=True)
        moved = elsewhere / "feature_reference.parquet"
        moved.write_bytes((directory / "feature_reference.parquet").read_bytes())
        (directory / "feature_reference.parquet").unlink()
        _edit_manifest(
            model_id,
            lambda payload: payload["monitoring"].update(
                feature_reference_uri=str(moved)
            ),
        )
        pd.testing.assert_frame_equal(load_monitoring_reference(model_id)[1], expected)

    def test_a_path_outside_this_runs_root_is_refused_rather_than_followed(
        self, patched_multi_factory, tmp_path
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_uri_outside_the_root"
        )
        directory = _artifacts.run_dir(model_id)
        outside = tmp_path / "not_the_runs_root"
        outside.mkdir(parents=True, exist_ok=True)
        smuggled = outside / "feature_reference.parquet"
        smuggled.write_bytes((directory / "feature_reference.parquet").read_bytes())
        (directory / "feature_reference.parquet").unlink()
        _edit_manifest(
            model_id,
            lambda payload: payload["monitoring"].update(
                feature_reference_uri=str(smuggled)
            ),
        )
        with pytest.raises(ValidationError, match="not in the model's directory"):
            load_monitoring_reference(model_id)

    def test_a_named_reference_that_is_absent_is_a_broken_package(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_reference_named_but_absent"
        )
        _edit_manifest(
            model_id,
            lambda payload: payload["monitoring"].update(
                feature_reference_uri="feature_reference_that_never_arrived.parquet"
            ),
        )
        with pytest.raises(ValidationError, match="re-pulled") as excinfo:
            load_monitoring_reference(model_id)
        message = str(excinfo.value)
        assert "feature_reference_that_never_arrived.parquet" in message
        assert str(_artifacts.run_dir(model_id)) in message

    def test_a_manifest_with_no_reference_at_all_is_the_old_model(
        self, patched_multi_factory
    ):
        """The one case that is still (profile, None, None): a model
        registered before references were kept, which is what the retrain
        advice was written for."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_registered_before_references"
        )

        def _drop_the_references(payload):
            payload["monitoring"].pop("feature_reference_uri")
            payload["monitoring"].pop("prediction_reference_uri")

        _edit_manifest(model_id, _drop_the_references)
        profile, features, predictions = load_monitoring_reference(model_id)
        assert features is None and predictions is None
        assert set(profile["features"]) == set(load_manifest(model_id).feature_ids)

    def test_the_resolver_names_the_model_and_the_file_it_wanted(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_resolver_message"
        )
        assert resolve_model_artifact(model_id, "oos_predictions.parquet") == (
            _artifacts.run_dir(model_id) / "oos_predictions.parquet"
        )
        with pytest.raises(ValidationError, match="is not in the model's directory"):
            resolve_model_artifact(model_id, "a_file_this_model_never_had.parquet")


class TestTheMirrorIsWrittenUnderTheModelId:
    def test_mirroring_under_another_prefix_is_not_an_option(
        self, patched_multi_factory
    ):
        """A package written under any other prefix is one nothing can
        list and nothing can pull."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_mirror_prefix_is_gone"
        )
        store = store_from_url(_memory_url())
        with pytest.raises(TypeError):
            mirror_model_package(model_id, store, prefix="some_other_prefix")
        uris = mirror_model_package(model_id, store)
        assert all(f"{model_id}/" in uri for uri in uris.values())


class TestTheStoreThroughTheToolSurface:
    def test_a_registration_is_listed_pulled_and_arrives_validated(
        self, patched_multi_factory, monkeypatch, tmp_path
    ):
        url = _memory_url()
        monkeypatch.setenv(MIRROR_URL_ENV, url)
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_store_round_trip"
        )
        # The promotion log travels with the package, and is pushed to the
        # mirror as it is recorded.
        promote(model_id, "validated", "rank IC held on every fold")

        listing = list_remote_models(ListRemoteModelsInput())
        assert listing.store_url == url
        assert listing.models == [model_id] and listing.n_total == 1
        assert listing.warnings == []

        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "another_root"))
        result = pull_model_package(PullModelPackageInput(model_id=model_id))
        assert result.ok and result.model_id == model_id
        assert result.stage == "validated"
        assert result.store_url == url
        assert result.mismatched == [] and result.missing == []
        assert "model.joblib" in result.verified
        assert result.signature is None and result.key_pinned is False
        assert any("UNSIGNED" in w for w in result.warnings)
        assert result.registry_dir == str(_artifacts.run_dir(model_id))
        assert load_manifest(model_id).model_id == model_id

        # A second pull is a refusal, because a registered model's
        # directory is the evidence behind everything recorded against it.
        with pytest.raises(ValidationError, match="overwrite=True"):
            pull_model_package(PullModelPackageInput(model_id=model_id))
        assert pull_model_package(
            PullModelPackageInput(model_id=model_id, overwrite=True)
        ).ok

    def test_a_listing_says_how_many_it_did_not_return(
        self, patched_multi_factory, monkeypatch
    ):
        url = _memory_url()
        monkeypatch.setenv(MIRROR_URL_ENV, url)
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_truncated_listing"
        )
        store = store_from_url(url)
        # A second package in the store, listed by its manifest key the
        # same way the first one is.
        store.put("mdl_secondpackage/manifest.json", b"{}")

        listing = list_remote_models(ListRemoteModelsInput(limit=1))
        assert listing.n_total == 2 and len(listing.models) == 1
        assert any("2 models in the store" in w for w in listing.warnings)
        full = list_remote_models(ListRemoteModelsInput(store_url=url))
        assert full.models == sorted([model_id, "mdl_secondpackage"])
        assert full.warnings == []

    def test_an_empty_store_says_so_rather_than_returning_nothing(self):
        result = list_remote_models(ListRemoteModelsInput(store_url=_memory_url()))
        assert result.models == [] and result.n_total == 0
        assert any("no model packages" in w for w in result.warnings)

    def test_no_store_anywhere_names_both_ways_to_give_one(self, monkeypatch):
        monkeypatch.delenv(MIRROR_URL_ENV, raising=False)
        for call in (
            lambda: list_remote_models(ListRemoteModelsInput()),
            lambda: pull_model_package(PullModelPackageInput(model_id="mdl_absent")),
        ):
            with pytest.raises(ValidationError) as excinfo:
                call()
            message = str(excinfo.value)
            assert "store_url" in message and MIRROR_URL_ENV in message

    def test_a_scheme_nothing_can_open_names_the_ones_that_work(self, monkeypatch):
        monkeypatch.delenv(MIRROR_URL_ENV, raising=False)
        with pytest.raises(ValidationError) as excinfo:
            list_remote_models(ListRemoteModelsInput(store_url="zzz://bucket/x"))
        message = str(excinfo.value)
        assert "file://" in message and "s3://" in message and "memory://" in message

    def test_an_unsigned_package_is_refused_when_a_signature_is_required(
        self, patched_multi_factory, monkeypatch, tmp_path
    ):
        url = _memory_url()
        monkeypatch.setenv(MIRROR_URL_ENV, url)
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_unsigned_but_required"
        )
        elsewhere = tmp_path / "another_root"
        monkeypatch.setenv("SQT_RUNS_DIR", str(elsewhere))
        with pytest.raises(ValidationError, match="not signed"):
            pull_model_package(
                PullModelPackageInput(model_id=model_id, require_signature=True)
            )
        assert not (elsewhere / model_id).exists()

    @pytest.mark.skipif(not signing_available(), reason="cryptography is not installed")
    def test_the_wrong_pinned_key_registers_nothing(
        self, patched_multi_factory, monkeypatch, tmp_path
    ):
        from standard_quant_tools.audit.signing import generate_keypair

        private, public = generate_keypair()
        _other_private, other_public = generate_keypair()
        key_path = tmp_path / "signing.key"
        key_path.write_bytes(private)
        trusted_path = tmp_path / "trusted.pub"
        trusted_path.write_bytes(public)
        other_path = tmp_path / "somebody_elses.pub"
        other_path.write_bytes(other_public)

        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_signed_package"
        )
        sign_manifest(model_id, key_path=key_path)
        url = _memory_url()
        mirror_model_package(model_id, store_from_url(url))

        elsewhere = tmp_path / "another_root"
        monkeypatch.setenv("SQT_RUNS_DIR", str(elsewhere))
        with pytest.raises(ValidationError, match="not the pinned"):
            pull_model_package(
                PullModelPackageInput(
                    model_id=model_id,
                    store_url=url,
                    require_signature=True,
                    public_key_path=str(other_path),
                )
            )
        assert not (elsewhere / model_id / "manifest.json").exists()
        with pytest.raises(ValidationError, match="no registered model"):
            load_manifest(model_id)

        result = pull_model_package(
            PullModelPackageInput(
                model_id=model_id,
                store_url=url,
                require_signature=True,
                public_key_path=str(trusted_path),
            )
        )
        assert result.ok and result.key_pinned is True
        assert result.signature["public_key"] == public.hex()
        assert not any("UNSIGNED" in w for w in result.warnings)

    def test_a_nested_remote_key_is_refused_and_registers_nothing(
        self, patched_multi_factory, monkeypatch, tmp_path
    ):
        """A store is flat, one run directory deep, like the runs root
        itself. The key is written through the filesystem underneath the
        store, because the store's own put refuses it -- which is the
        point: a store somebody else wrote to is not bound by our
        validation, and the pull has to be."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_nested_remote_key"
        )
        url = _memory_url()
        store = store_from_url(url)
        mirror_model_package(model_id, store)
        with pytest.raises(ValidationError, match="segments"):
            store.put(f"{model_id}/sub/evil.bin", b"payload")
        store._fs.pipe_file(f"{store._root}/{model_id}/sub/evil.bin", b"payload")

        elsewhere = tmp_path / "another_root"
        monkeypatch.setenv("SQT_RUNS_DIR", str(elsewhere))
        with pytest.raises(ValidationError, match="a store is flat"):
            pull_model_package(PullModelPackageInput(model_id=model_id, store_url=url))
        assert not (elsewhere / model_id / "manifest.json").exists()
        assert not Path(elsewhere / model_id / "sub").exists()
        with pytest.raises(ValidationError, match="no registered model"):
            load_manifest(model_id)
