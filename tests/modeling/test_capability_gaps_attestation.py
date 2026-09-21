"""
Attestation, and the promotion that has to rest on one.

The content hashes in a manifest catch an edited artifact, and every
loader checks the one file it is about to read. What they cannot catch is
a manifest rewritten together with the artifact it describes -- the
manifest is the root of those hashes and cannot contain its own -- and
the Ed25519 signature over the manifest bytes is the only check that
does. The downgrade is planted here in the form that is easiest to
perform and was previously invisible through the tools: edit the
manifest, then DELETE the signature rather than forging one. The bare
library verification calls that clean, which is exactly why the tool
requires a signature by default, and both halves of the contrast are
pinned below so neither can drift without the other being noticed.

The promotion gate is the second half. A stage is a statement that
somebody read the evidence, so a model whose `model.joblib` no longer
hashes to its manifest is refused the stage; waiving the check is
allowed and lands in the promotion's own evidence, where a reader months
later can see it. See the CHANGELOG entry of 2026-09-21.
"""

import hashlib
import json

import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import (
    AttestModelPackageInput,
    InspectModelInput,
    PromoteModelInput,
)
from standard_quant_tools.modeling.agent.tools import (
    attest_model_package,
    inspect_model,
    promote_model,
)
from standard_quant_tools.modeling.registry.lifecycle import current_stage, promotions
from standard_quant_tools.modeling.registry.package import verify_model_package
from standard_quant_tools.modeling.registry.signing import (
    SIGNATURE_FILE,
    VERIFY_KEY_ENV,
    sign_manifest,
    signing_available,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec

needs_signing = pytest.mark.skipif(
    not signing_available(), reason="cryptography is not installed"
)


@pytest.fixture(autouse=True)
def _nothing_pinned_by_the_environment(monkeypatch):
    """A verification key left in the environment by another test would
    turn every unpinned case below into a pinned one silently."""
    monkeypatch.delenv(VERIFY_KEY_ENV, raising=False)


@pytest.fixture
def keypair(tmp_path):
    from standard_quant_tools.audit.signing import generate_keypair

    private, public = generate_keypair()
    key_path = tmp_path / "signing.key"
    key_path.write_bytes(private)
    pub_path = tmp_path / "signing.pub"
    pub_path.write_bytes(public)
    return key_path, pub_path, public


@pytest.fixture
def other_public_key_path(tmp_path):
    from standard_quant_tools.audit.signing import generate_keypair

    _, public = generate_keypair()
    path = tmp_path / "somebody_elses.pub"
    path.write_bytes(public)
    return path


def _manifest_path(model_id: str):
    return _artifacts.run_dir(model_id) / "manifest.json"


def _manifest_sha256(model_id: str) -> str:
    return hashlib.sha256(_manifest_path(model_id).read_bytes()).hexdigest()


def _rewrite_manifest(model_id: str, **changes) -> None:
    """Edit the manifest in place, leaving it a manifest: the point of
    every case here is a package that still parses and still passes the
    content hashes, because those are the checks that cannot see this."""
    path = _manifest_path(model_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


class TestAttestation:
    @needs_signing
    def test_a_deleted_signature_reads_clean_to_the_library_and_not_to_the_tool(
        self, patched_multi_factory, keypair
    ):
        """Deleting a signature is easier than forging one.

        The manifest is edited to claim a headline R2 nobody measured and
        the signature is deleted rather than replaced. Every content hash
        still passes -- the manifest is not one of the files it hashes --
        so the library's default verification says the package is fine,
        and that is the whole gap the tool's default closes.
        """
        key_path, _, _ = keypair
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_signature_downgrade"
        )
        sign_manifest(model_id, key_path=key_path)
        manifest = json.loads(_manifest_path(model_id).read_text(encoding="utf-8"))
        _rewrite_manifest(
            model_id, oos_metrics={**(manifest["oos_metrics"] or {}), "r2": 0.91}
        )
        (_artifacts.run_dir(model_id) / SIGNATURE_FILE).unlink()

        bare = verify_model_package(model_id)
        assert bare.ok is True
        assert bare.signature is None and bare.signature_error is None
        assert not bare.mismatched and not bare.missing

        attested = attest_model_package(AttestModelPackageInput(model_id=model_id))
        assert attested.ok is False
        assert attested.signature_error is not None
        assert "is not signed" in attested.signature_error
        assert attested.key_pinned is False
        assert attested.signature is None
        # The digest is still reported: it is what a later decision would
        # have to name, and it is not the one that was signed.
        assert attested.manifest_sha256 == _manifest_sha256(model_id)

    @needs_signing
    def test_a_signed_package_under_the_pinned_key_attests(
        self, patched_multi_factory, keypair
    ):
        key_path, pub_path, public = keypair
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_attest_pinned"
        )
        sign_manifest(model_id, key_path=key_path)
        result = attest_model_package(
            AttestModelPackageInput(model_id=model_id, public_key_path=str(pub_path))
        )
        assert result.ok is True
        assert result.key_pinned is True
        assert result.signature_error is None
        assert result.signature["public_key"] == public.hex()
        assert result.manifest_sha256 == _manifest_sha256(model_id)
        assert result.signature["manifest_sha256"] == result.manifest_sha256
        assert "model.joblib" in result.verified
        # What the hashes do not vouch for is named rather than implied.
        assert SIGNATURE_FILE in result.unhashed
        assert result.warnings == []

    @needs_signing
    def test_the_wrong_pinned_key_is_a_finding_not_a_pass(
        self, patched_multi_factory, keypair, other_public_key_path
    ):
        key_path, _, _ = keypair
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_attest_wrong_key"
        )
        sign_manifest(model_id, key_path=key_path)
        result = attest_model_package(
            AttestModelPackageInput(
                model_id=model_id, public_key_path=str(other_public_key_path)
            )
        )
        assert result.ok is False
        assert result.key_pinned is False
        assert "not the pinned" in result.signature_error
        assert result.signature is None

    @needs_signing
    def test_an_unpinned_signature_says_what_it_did_not_establish(
        self, patched_multi_factory, keypair
    ):
        key_path, _, public = keypair
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_attest_unpinned"
        )
        sign_manifest(model_id, key_path=key_path)
        result = attest_model_package(AttestModelPackageInput(model_id=model_id))
        assert result.ok is True
        assert result.key_pinned is False
        assert result.signature["public_key"] == public.hex()
        assert any(
            "not that anyone you trust wrote them" in warning
            for warning in result.warnings
        )
        assert any(VERIFY_KEY_ENV in warning for warning in result.warnings)

    def test_the_hashes_alone_can_be_attested_and_say_so(self, patched_multi_factory):
        """A shop with no signing key still gets an answer, and the answer
        says which question it answered."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_attest_hashes_only"
        )
        result = attest_model_package(
            AttestModelPackageInput(model_id=model_id, require_signature=False)
        )
        assert result.ok is True
        assert result.signature is None and result.signature_error is None
        assert result.key_pinned is False
        assert result.manifest_sha256 == _manifest_sha256(model_id)
        assert any("unsigned" in warning for warning in result.warnings)

    def test_a_swapped_artifact_is_named_by_the_attestation(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_attest_swapped"
        )
        with open(_artifacts.run_dir(model_id) / "model.joblib", "ab") as handle:
            handle.write(b"\x00")
        result = attest_model_package(
            AttestModelPackageInput(model_id=model_id, require_signature=False)
        )
        assert result.ok is False
        assert result.mismatched == ["model.joblib"]
        assert any("not the one that was registered" in w for w in result.warnings)


class TestPromotionRestsOnTheEvidence:
    def test_a_swapped_binary_stops_the_promotion_and_leaves_no_record(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_promote_swapped"
        )
        with open(_artifacts.run_dir(model_id) / "model.joblib", "ab") as handle:
            handle.write(b"\x00")
        with pytest.raises(ValidationError, match="does not verify"):
            promote_model(
                PromoteModelInput(
                    model_id=model_id,
                    to_stage="validated",
                    reason="the folds agreed on the sign",
                )
            )
        assert promotions(model_id) == []
        assert current_stage(model_id) == "candidate"

    def test_the_check_can_be_waived_and_the_waiver_is_in_the_record(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_promote_waived"
        )
        with open(_artifacts.run_dir(model_id) / "model.joblib", "ab") as handle:
            handle.write(b"\x00")
        result = promote_model(
            PromoteModelInput(
                model_id=model_id,
                to_stage="validated",
                reason="the binary was rebuilt by hand and we accept it",
                require_verified_package=False,
            )
        )
        assert result.to_stage == "validated"
        assert result.package_ok is False
        recorded = promotions(model_id)[-1].evidence
        assert any(
            entry == f"manifest_sha256={_manifest_sha256(model_id)}"
            for entry in recorded
        )
        assert any("package_check_waived" in entry for entry in recorded)
        assert any("model.joblib" in entry for entry in recorded)

    def test_a_clean_promotion_records_the_manifest_it_rested_on(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_promote_clean"
        )
        result = promote_model(
            PromoteModelInput(
                model_id=model_id,
                to_stage="validated",
                reason="rank IC held on every fold",
                evidence=["sqt://runs/ds_promote_clean/panel"],
            )
        )
        assert result.package_ok is True
        assert result.manifest_sha256 == _manifest_sha256(model_id)
        recorded = promotions(model_id)[-1].evidence
        assert recorded[0] == f"manifest_sha256={result.manifest_sha256}"
        assert recorded[1].startswith("package_verified=")
        assert int(recorded[1].split("=")[1].split()[0]) > 0
        # What the caller passed is kept, after what the gate found.
        assert recorded[-1] == "sqt://runs/ds_promote_clean/panel"
        # The manifest is still untouched by a promotion.
        assert _manifest_sha256(model_id) == result.manifest_sha256

    def test_requiring_a_signature_refuses_an_unsigned_package(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_promote_unsigned"
        )
        with pytest.raises(ValidationError, match="does not verify"):
            promote_model(
                PromoteModelInput(
                    model_id=model_id,
                    to_stage="validated",
                    reason="production here only takes signed packages",
                    require_signature=True,
                )
            )
        assert current_stage(model_id) == "candidate"
        # Without that bar, the same package promotes.
        assert (
            promote_model(
                PromoteModelInput(
                    model_id=model_id,
                    to_stage="validated",
                    reason="production here only takes signed packages",
                )
            ).to_stage
            == "validated"
        )

    @needs_signing
    def test_a_signed_package_promotes_under_its_key(
        self, patched_multi_factory, keypair, other_public_key_path
    ):
        key_path, pub_path, _ = keypair
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_promote_signed"
        )
        sign_manifest(model_id, key_path=key_path)
        with pytest.raises(ValidationError, match="not the pinned"):
            promote_model(
                PromoteModelInput(
                    model_id=model_id,
                    to_stage="validated",
                    reason="signed, but not by the key we trust",
                    require_signature=True,
                    public_key_path=str(other_public_key_path),
                )
            )
        assert promotions(model_id) == []
        result = promote_model(
            PromoteModelInput(
                model_id=model_id,
                to_stage="validated",
                reason="signed by the key production pins",
                require_signature=True,
                public_key_path=str(pub_path),
            )
        )
        assert result.package_ok is True
        assert result.manifest_sha256 == _manifest_sha256(model_id)

    def test_a_model_registered_before_content_hashing_still_promotes(
        self, patched_multi_factory
    ):
        """An empty `content_hashes` is an old package, not a broken one.

        Nothing is claimed about its files, so nothing about them can
        fail; the gate defaults to on precisely because it cannot lock
        out the models that predate the hashes.
        """
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_promote_prehash"
        )
        _rewrite_manifest(model_id, content_hashes={})
        result = promote_model(
            PromoteModelInput(
                model_id=model_id,
                to_stage="validated",
                reason="registered before the package was content hashed",
            )
        )
        assert result.package_ok is True
        assert promotions(model_id)[-1].evidence[1] == "package_verified=0 files"


class TestLineageTakesAKey:
    @needs_signing
    def test_the_lineage_view_reports_the_pinned_key_error(
        self, patched_multi_factory, keypair, other_public_key_path
    ):
        key_path, _, _ = keypair
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_lineage_key"
        )
        sign_manifest(model_id, key_path=key_path)
        wrong = inspect_model(
            InspectModelInput(
                model_id=model_id,
                view="lineage",
                public_key_path=str(other_public_key_path),
            )
        ).data
        assert wrong["package"]["ok"] is False
        assert "not the pinned" in wrong["package"]["signature_error"]

        unpinned = inspect_model(
            InspectModelInput(model_id=model_id, view="lineage")
        ).data
        # Without a key the view is the one it always was: same fields,
        # and a signature checked against the key it carries.
        assert set(unpinned) == set(wrong)
        assert set(unpinned["package"]) == set(wrong["package"])
        assert unpinned["package"]["ok"] is True
        assert unpinned["package"]["signature"]["key_pinned"] is False

    def test_a_key_with_nothing_to_check_is_refused_not_ignored(
        self, patched_multi_factory, other_public_key_path
    ):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_lineage_key_misplaced"
        )
        with pytest.raises(PydanticValidationError, match="nothing to check"):
            InspectModelInput(
                model_id=model_id,
                view="summary",
                public_key_path=str(other_public_key_path),
            )
