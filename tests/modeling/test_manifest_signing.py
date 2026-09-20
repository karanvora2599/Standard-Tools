"""
Signed manifests: authenticity on top of the content hashes.

The case the signature exists for is planted directly: the manifest is
edited without touching the signature, every content hash still passes
(the manifest is the root of those hashes and cannot contain its own),
and the signature is what refuses. The wrong pinned key is refused by
name, because a valid signature under an unknown key establishes less
than it looks like it does, and the report says which of the two it
established.
"""

import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.registry.package import verify_model_package
from standard_quant_tools.modeling.registry.signing import (
    SIGNATURE_FILE,
    SIGNING_KEY_ENV,
    VERIFY_KEY_ENV,
    sign_manifest,
    signing_available,
    signing_configured,
    verify_manifest_signature,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec

pytestmark = pytest.mark.skipif(
    not signing_available(), reason="cryptography is not installed"
)


@pytest.fixture
def keypair(tmp_path):
    from standard_quant_tools.audit.signing import generate_keypair

    private, public = generate_keypair()
    key_path = tmp_path / "model_signing.key"
    key_path.write_bytes(private)
    pub_path = tmp_path / "model_signing.pub"
    pub_path.write_bytes(public)
    return key_path, pub_path, public


class TestSigning:
    def test_a_signature_verifies_and_says_what_it_established(
        self, patched_multi_factory, keypair
    ):
        key_path, pub_path, public = keypair
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_sign")
        with pytest.raises(ValidationError, match="not signed"):
            verify_manifest_signature(model_id)
        with pytest.raises(ValidationError, match="not signed"):
            load_manifest(model_id, require_signature=True)
        assert (
            load_manifest(model_id).model_id == model_id
        )  # unsigned is fine by default

        record = sign_manifest(model_id, key_path=key_path)
        assert record["public_key"] == public.hex()
        assert (_artifacts.run_dir(model_id) / SIGNATURE_FILE).exists()

        unpinned = verify_manifest_signature(model_id)
        assert unpinned["key_pinned"] is False
        assert unpinned["public_key"] == public.hex()
        assert (
            verify_manifest_signature(model_id, public_key=public)["key_pinned"] is True
        )
        assert verify_manifest_signature(model_id, public_key=str(pub_path))[
            "key_pinned"
        ]
        assert verify_manifest_signature(model_id, public_key=public.hex())[
            "key_pinned"
        ]
        assert load_manifest(model_id, require_signature=True).model_id == model_id

        report = verify_model_package(
            model_id, require_signature=True, public_key=public
        )
        assert report.ok and report.signature["key_pinned"] is True
        # The signature is not covered by the hashes it signs.
        assert SIGNATURE_FILE in report.unhashed

    def test_the_wrong_key_and_an_edited_manifest_are_refused_by_name(
        self, patched_multi_factory, keypair, tmp_path, monkeypatch
    ):
        from standard_quant_tools.audit.signing import generate_keypair

        key_path, _, _ = keypair
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_wrongkey")
        sign_manifest(model_id, key_path=key_path)
        _, other_public = generate_keypair()
        with pytest.raises(ValidationError, match="not the pinned"):
            verify_manifest_signature(model_id, public_key=other_public)
        other_path = tmp_path / "other.pub"
        other_path.write_bytes(other_public)
        monkeypatch.setenv(VERIFY_KEY_ENV, str(other_path))
        with pytest.raises(ValidationError, match="not the pinned"):
            load_manifest(model_id, require_signature=True)
        monkeypatch.delenv(VERIFY_KEY_ENV)

        # The manifest is edited and the signature is not: every content
        # hash still passes, and the signature is what refuses.
        path = _artifacts.run_dir(model_id) / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
        with pytest.raises(ValidationError, match="changed since it was signed"):
            verify_manifest_signature(model_id)
        report = verify_model_package(model_id)
        assert not report.mismatched and not report.missing
        assert report.signature_error and "changed" in report.signature_error
        assert not report.ok

    def test_a_garbled_signature_record_is_named(self, patched_multi_factory, keypair):
        key_path, _, _ = keypair
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_garbled")
        sign_manifest(model_id, key_path=key_path)
        (_artifacts.run_dir(model_id) / SIGNATURE_FILE).write_text("{not json", "utf-8")
        with pytest.raises(ValidationError, match="not a signature record"):
            verify_manifest_signature(model_id)

    def test_registration_signs_when_the_key_is_configured(
        self, patched_multi_factory, keypair, monkeypatch
    ):
        key_path, _, public = keypair
        assert signing_configured() is False
        monkeypatch.setenv(SIGNING_KEY_ENV, str(key_path))
        assert signing_configured() is True
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_autosign")
        assert verify_manifest_signature(model_id, public_key=public)["key_pinned"]
        assert verify_model_package(
            model_id, require_signature=True, public_key=public
        ).ok

    def test_a_signer_callback_needs_its_public_key(
        self, patched_multi_factory, keypair
    ):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key_path, _, public = keypair
        private = Ed25519PrivateKey.from_private_bytes(key_path.read_bytes())
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_signer")
        with pytest.raises(ValidationError, match="no signing key"):
            sign_manifest(model_id)
        with pytest.raises(ValidationError, match="public key"):
            sign_manifest(model_id, signer=private.sign)
        sign_manifest(model_id, signer=private.sign, public_key=public)
        assert verify_manifest_signature(model_id, public_key=public)["key_pinned"]
