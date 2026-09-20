"""
Signed manifests: authenticity on top of integrity.

WHAT THE HASHES COULD NOT SAY. Every artifact in a model package is
hashed into `manifest.json`, and every loader verifies before it reads --
`model.joblib` before it is deserialized, which is the ordering that
turns a swapped binary from arbitrary code execution into a refusal. But
the manifest is the root of that trust and cannot contain its own digest,
so anything that can rewrite BOTH an artifact and the manifest passes
every check. That was stated in the registry's own docstring as the
next step before it crosses a trust boundary. This is that step.

WHAT A SIGNATURE SAYS. `manifest.sig` is an Ed25519 signature over the
exact bytes of `manifest.json`, with the public key that made it. A
verifier holding a PINNED public key learns that the manifest -- and
through its hashes, every artifact -- is the one that key's holder
registered. A verifier with no pinned key learns less: that the manifest
and the signature were written together by whoever holds the embedded
key, which rules out an edit to the manifest alone but not an attacker
who rewrote both under a key of their own. The report says which of the
two it established (`key_pinned`), because the difference is the whole
point.

WRITE ORDER. The manifest is still written last and is still the commit
point; the signature is written after it, as an attestation on a package
that already exists. A model registered with `SQT_MODEL_SIGNING_KEY_PATH`
set is signed at registration; one registered without can be signed
later, and `load_manifest(..., require_signature=True)` and
`verify_model_package` are where a caller says that unsigned is not good
enough. Key custody is not this library's problem, exactly as
`audit/signing.py` says: a raw key file is for development, and a real
deployment passes a `signer` callback routed through its HSM or KMS with
the public key it corresponds to.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from standard_quant_tools.audit import signing as _audit_signing
from standard_quant_tools.error import ValidationError

from .. import artifacts as _artifacts

SIGNING_KEY_ENV = "SQT_MODEL_SIGNING_KEY_PATH"
VERIFY_KEY_ENV = "SQT_MODEL_VERIFY_KEY_PATH"
SIGNATURE_FILE = "manifest.sig"
ALGORITHM = "ed25519"
_RAW_KEY_BYTES = 32


def signing_available() -> bool:
    """Whether the optional `cryptography` dependency is importable."""
    return bool(_audit_signing.HAS_CRYPTOGRAPHY)


def signing_configured() -> bool:
    """Whether registrations in this process will be signed."""
    return bool(os.environ.get(SIGNING_KEY_ENV))


def _require() -> None:
    if not signing_available():
        raise ValidationError(
            "manifest signing needs the `cryptography` package "
            "(`pip install standard_quant_tools[signing]`)."
        )


def _manifest_bytes(model_id: str) -> "tuple[Path, bytes]":
    directory = _artifacts.run_dir(model_id)
    path = directory / "manifest.json"
    if not path.exists():
        raise ValidationError(f"no registered model with model_id={model_id!r}")
    return directory, path.read_bytes()


def _public_key_bytes(value: Union[bytes, str, Path, None]) -> Optional[bytes]:
    """A raw 32-byte Ed25519 public key from bytes, hex, a path, or the
    `SQT_MODEL_VERIFY_KEY_PATH` environment variable; None when nothing
    is pinned."""
    if value is None:
        env_path = os.environ.get(VERIFY_KEY_ENV)
        if not env_path:
            return None
        value = Path(env_path)
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, Path):
        if not value.exists():
            raise ValidationError(f"verification key file not found: {value}")
        raw = value.read_bytes()
    elif isinstance(value, str) and len(value) == 2 * _RAW_KEY_BYTES:
        try:
            raw = bytes.fromhex(value)
        except ValueError as exc:
            raise ValidationError(f"public key {value!r} is not hex") from exc
    else:
        path = Path(str(value))
        if not path.exists():
            raise ValidationError(f"verification key file not found: {path}")
        raw = path.read_bytes()
    if len(raw) != _RAW_KEY_BYTES:
        raise ValidationError(
            f"an Ed25519 public key is {_RAW_KEY_BYTES} raw bytes; got {len(raw)}."
        )
    return raw


def sign_manifest(
    model_id: str,
    *,
    key_path: Optional[Union[str, Path]] = None,
    signer: Optional[Callable[[bytes], bytes]] = None,
    public_key: Union[bytes, str, Path, None] = None,
) -> Dict[str, Any]:
    """
    Write `manifest.sig` beside a registered model's manifest.

    The key comes from `key_path`, else `SQT_MODEL_SIGNING_KEY_PATH`, as a
    raw Ed25519 private key file (`sqt keygen` makes one for development).
    A deployment that never lets a private key touch disk passes `signer`
    -- a callback that returns the signature over the bytes it is given --
    together with the `public_key` it corresponds to, which the signature
    record carries so a verifier can name who signed.
    """
    _require()
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    directory, payload = _manifest_bytes(model_id)
    if signer is not None:
        embedded = _public_key_bytes(public_key)
        if embedded is None:
            raise ValidationError(
                "sign_manifest: a signer callback needs the public key it "
                "corresponds to (public_key=...), so the signature record can "
                "say which key signed."
            )
        sign = signer
    else:
        path = Path(key_path) if key_path else None
        if path is None and os.environ.get(SIGNING_KEY_ENV):
            path = Path(os.environ[SIGNING_KEY_ENV])
        if path is None or not path.exists():
            raise ValidationError(
                f"sign_manifest: no signing key. Pass key_path=..., set "
                f"{SIGNING_KEY_ENV}, or pass a signer callback (routed through "
                "an HSM/KMS) with its public key. `sqt keygen` writes a "
                "development keypair."
            )
        private = Ed25519PrivateKey.from_private_bytes(path.read_bytes())
        embedded = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        sign = private.sign
    record = {
        "algorithm": ALGORITHM,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "public_key": embedded.hex(),
        "signature": bytes(sign(payload)).hex(),
        "signed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _artifacts._atomic_write_bytes(
        directory / SIGNATURE_FILE,
        json.dumps(record, indent=2, sort_keys=True).encode("utf-8"),
    )
    return record


def verify_manifest_signature(
    model_id: str,
    *,
    public_key: Union[bytes, str, Path, None] = None,
) -> Dict[str, Any]:
    """
    Check `manifest.sig` against the manifest's current bytes.

    With a pinned `public_key` (or `SQT_MODEL_VERIFY_KEY_PATH`), a
    signature under any other key is refused by name. Without one, the
    signature is checked against the key it carries and the report says
    `key_pinned: False`, which is the honest description of what that
    established. Refused, never False: an unsigned model, a signature
    that does not verify, a signature under the wrong key and an
    unreadable record are four different findings and each is named.
    """
    directory, payload = _manifest_bytes(model_id)
    sig_path = directory / SIGNATURE_FILE
    if not sig_path.exists():
        raise ValidationError(
            f"model {model_id!r} is not signed: no {SIGNATURE_FILE} beside its "
            f"manifest. Sign it with sign_manifest, or register with "
            f"{SIGNING_KEY_ENV} set."
        )
    return verify_signature_bytes(
        payload, sig_path.read_bytes(), public_key=public_key, model_id=model_id
    )


def verify_signature_bytes(
    payload: bytes,
    record_bytes: bytes,
    *,
    public_key: Union[bytes, str, Path, None] = None,
    model_id: str = "?",
) -> Dict[str, Any]:
    """
    The check itself, on bytes: the manifest as read and the signature
    record beside it. `verify_manifest_signature` reads a registered
    package and calls this; `pull_model_package` calls it on what a
    store returned BEFORE writing any of it locally.
    """
    _require()
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        record = json.loads(record_bytes.decode("utf-8"))
        algorithm = record["algorithm"]
        embedded = bytes.fromhex(record["public_key"])
        signature = bytes.fromhex(record["signature"])
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValidationError(
            f"{SIGNATURE_FILE} for model {model_id!r} is not a signature "
            f"record: {exc}"
        ) from exc
    if algorithm != ALGORITHM:
        raise ValidationError(
            f"model {model_id!r} is signed with {algorithm!r}; only "
            f"{ALGORITHM!r} is verified here."
        )
    pinned = _public_key_bytes(public_key)
    if pinned is not None and pinned != embedded:
        raise ValidationError(
            f"model {model_id!r} was signed by key {embedded.hex()[:16]}..., "
            "which is not the pinned verification key. A valid signature "
            "under an unknown key proves the manifest and the signature were "
            "written together, not that anyone you trust wrote them."
        )
    try:
        Ed25519PublicKey.from_public_bytes(embedded).verify(signature, payload)
    except (InvalidSignature, ValueError) as exc:
        raise ValidationError(
            f"the signature on model {model_id!r} does not verify: "
            "manifest.json has changed since it was signed, or "
            f"{SIGNATURE_FILE} is not a signature over this manifest."
        ) from exc
    return {
        "algorithm": ALGORITHM,
        "public_key": embedded.hex(),
        "key_pinned": pinned is not None,
        "signed_at_utc": record.get("signed_at_utc"),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


__all__ = [
    "ALGORITHM",
    "SIGNATURE_FILE",
    "SIGNING_KEY_ENV",
    "VERIFY_KEY_ENV",
    "sign_manifest",
    "signing_available",
    "signing_configured",
    "verify_manifest_signature",
    "verify_signature_bytes",
]
