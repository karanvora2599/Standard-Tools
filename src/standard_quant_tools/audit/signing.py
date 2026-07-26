"""
Ed25519 checkpoint signing -- an optional external anchor on top of the
hash chain. `verify_audit_log_integrity`/`verify_audit_trail_integrity`
explicitly cannot catch an attacker who consistently rewrites an entire day
file *and* its chain-index entry to stay internally self-consistent (see
`verify.py`'s docstrings) -- there is no anchor outside those files
themselves. A signed checkpoint is that anchor: `verify_checkpoint_signature`
only needs the public key, not any trust in the JSONL files' own internal
consistency.

`cryptography` is an optional dependency
(`pip install standard_quant_tools[signing]`) -- every other part of the
audit trail works without it. Calling anything in this module without it
installed raises a clear `ImportError` with install instructions, the same
"graceful, explicit failure" contract `data.bloomberg_provider` uses for
`blpapi` and `_sqt_core`'s C++ extension uses elsewhere in this codebase.

Key custody is explicitly NOT this library's problem. `SQT_AUDIT_SIGNING_KEY_PATH`
(or an explicit `key_path`) points at a raw Ed25519 private key file for
local development -- `generate_keypair()` / `sqt keygen` make one, and are
labeled for that purpose only, not production key custody. A real
deployment should pass its own `signer: Callable[[bytes], bytes]` instead,
routed through an HSM/KMS, and never let this library see a bare private
key at all.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from .paths import _INDEX_FILENAME, _audit_dir

HAS_CRYPTOGRAPHY = False
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    HAS_CRYPTOGRAPHY = True
except ImportError:
    pass


def _require_cryptography() -> None:
    if not HAS_CRYPTOGRAPHY:
        raise ImportError(
            "cryptography is not installed. Ed25519 checkpoint signing "
            "requires it, and it isn't a hard dependency of this package. "
            "Install it with `pip install standard_quant_tools[signing]` "
            "(or `pip install cryptography` directly)."
        )


def generate_keypair() -> Tuple[bytes, bytes]:
    """
    Generate a new Ed25519 keypair, returned as `(private_bytes, public_bytes)`
    in raw encoding. For local development only — **not** a production key
    custody solution. A real deployment should generate/store keys through
    its own KMS/HSM and pass a `signer` callback to `checkpoint_and_sign`
    instead of ever writing a bare private key file.
    """
    _require_cryptography()
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private_bytes, public_bytes


def _load_signer(key_path: Optional[Union[str, Path]]) -> Callable[[bytes], bytes]:
    """
    Resolve a signing callback from a raw Ed25519 private key file: the
    explicit `key_path` param if given, else `SQT_AUDIT_SIGNING_KEY_PATH`.
    Raises a clear error if neither resolves to an existing file — the
    caller should pass their own `signer` callback instead if they don't
    want a bare key file on disk at all.
    """
    _require_cryptography()
    path = Path(key_path) if key_path else None
    if path is None:
        env_path = os.environ.get("SQT_AUDIT_SIGNING_KEY_PATH")
        path = Path(env_path) if env_path else None
    if path is None or not path.exists():
        raise FileNotFoundError(
            "No signing key found. Pass key_path=..., set "
            "SQT_AUDIT_SIGNING_KEY_PATH, or pass your own `signer` callback "
            "(e.g. routed through an HSM/KMS) instead of a bare key file."
        )
    private_key = Ed25519PrivateKey.from_private_bytes(path.read_bytes())

    def _sign(payload: bytes) -> bytes:
        return private_key.sign(payload)

    return _sign


def _derive_checkpoint_content(date: str, audit_dir: Path) -> Dict[str, Any]:
    """The content-derived half of a checkpoint: the day file's last
    record_hash and the chain index's index_hash entry for that date, as
    they currently stand on disk. Excludes `date`/`signed_at_utc`, which are
    metadata rather than something re-derivable from current state — kept
    separate so `verify_checkpoint_signature` can re-derive just this part
    and compare it against what was signed, without a always-fresh
    timestamp making every comparison fail."""
    from .writer import AuditWriter

    day_path = audit_dir / f"{date}.jsonl"
    index_path = audit_dir / _INDEX_FILENAME

    writer = AuditWriter(audit_dir=audit_dir)
    final_record_hash = writer._last_record_hash_in_file(day_path)

    index_hash = None
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("date") == date:
                    index_hash = entry.get("index_hash")

    return {"final_record_hash": final_record_hash, "index_hash": index_hash}


def checkpoint_and_sign(
    date: str,
    audit_dir: Optional[Union[str, Path]] = None,
    key_path: Optional[Union[str, Path]] = None,
    signer: Optional[Callable[[bytes], bytes]] = None,
) -> Path:
    """
    Build a checkpoint for `date` — `{date, final_record_hash, index_hash,
    signed_at_utc}` — and sign it with Ed25519, writing
    `<date>.checkpoint.json` (the checkpoint payload) and
    `<date>.checkpoint.sig` (the raw signature, hex-encoded) as sidecars in
    the audit directory. A periodic signed checkpoint, not a signature on
    every record, is enough: the hash chain already covers per-record
    integrity, the checkpoint anchors the chain's endpoint for that day.

    Signing key: pass `signer` (e.g. routed through an HSM/KMS) for
    anything beyond local development, OR `key_path` / the
    `SQT_AUDIT_SIGNING_KEY_PATH` env var pointing at a raw Ed25519 private
    key file (see `generate_keypair`/`sqt keygen` — development only).

    Returns the checkpoint JSON path.
    """
    _require_cryptography()
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    content = _derive_checkpoint_content(date, directory)
    checkpoint = {
        "date": date,
        "final_record_hash": content["final_record_hash"],
        "index_hash": content["index_hash"],
        "signed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    canonical = json.dumps(checkpoint, sort_keys=True).encode("utf-8")
    sign_fn = signer if signer is not None else _load_signer(key_path)
    signature = sign_fn(canonical)

    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"{date}.checkpoint.json"
    sig_path = directory / f"{date}.checkpoint.sig"
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8"
    )
    sig_path.write_text(signature.hex(), encoding="utf-8")
    return checkpoint_path


def verify_checkpoint_signature(
    date: str,
    public_key_path: Union[str, Path],
    audit_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Verify a checkpoint's signature using **only the public key** —
    independent of trusting the JSONL files' own internal consistency. Also
    re-derives the checkpoint's content from the current day file/index and
    confirms it still matches what was signed, so a checkpoint whose day
    file was altered *after* signing fails verification even though the
    signature over the (now stale) stored checkpoint is technically valid.

    Returns `False` (never raises) for: missing checkpoint/signature files,
    a public key that doesn't match the signing key, a corrupted signature,
    or a checkpoint that no longer matches the current on-disk state.
    """
    _require_cryptography()
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    checkpoint_path = directory / f"{date}.checkpoint.json"
    sig_path = directory / f"{date}.checkpoint.sig"
    if not checkpoint_path.exists() or not sig_path.exists():
        return False

    try:
        stored_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        signature = bytes.fromhex(sig_path.read_text(encoding="utf-8").strip())
        public_key = Ed25519PublicKey.from_public_bytes(
            Path(public_key_path).read_bytes()
        )

        canonical = json.dumps(stored_checkpoint, sort_keys=True).encode("utf-8")
        public_key.verify(signature, canonical)  # raises InvalidSignature on mismatch

        current_content = _derive_checkpoint_content(date, directory)
        if (
            stored_checkpoint.get("final_record_hash")
            != current_content["final_record_hash"]
            or stored_checkpoint.get("index_hash") != current_content["index_hash"]
        ):
            return False
        return True
    except Exception:
        return False
