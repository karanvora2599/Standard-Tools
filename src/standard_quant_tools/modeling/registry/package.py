"""
A model package as one thing: verify all of it, or copy all of it.

Each loader in `model_registry` verifies the one artifact it is about to
read, which is the right place for that check and leaves a question
nobody could answer without reading every artifact: does the package as
registered still exist, whole, and is it the one somebody signed. That
is a question asked at a trust boundary -- before a model is promoted,
before it is copied somewhere else, in the lineage view someone opens
months later -- and `verify_model_package` answers it in one pass.

`mirror_model_package` is the copy. It is written against the
`ArtifactStore` protocol rather than a path, so a verified package goes
to a bucket the same way it goes to a directory, and every hashed file
is re-hashed THROUGH THE TARGET after the copy: the digest that lands is
the digest that was registered, or the mirror refuses. The manifest is
copied last, so the commit-point property holds on the target too.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from standard_quant_tools._runspath import validate_identifier
from standard_quant_tools.artifact_store import ArtifactStore, LocalArtifactStore
from standard_quant_tools.error import ValidationError

from .lifecycle import PROMOTIONS_FILE
from .model_registry import load_manifest
from .signing import SIGNATURE_FILE, verify_manifest_signature

MANIFEST_FILE = "manifest.json"


@dataclass
class PackageVerification:
    """What one pass over a package found, file by file."""

    model_id: str
    #: Files whose bytes hash to what the manifest recorded.
    verified: List[str] = field(default_factory=list)
    #: Files present whose hash no longer matches.
    mismatched: List[str] = field(default_factory=list)
    #: Files the manifest hashes that are not there.
    missing: List[str] = field(default_factory=list)
    #: Files present that the manifest does not cover: the signature, the
    #: promotion log, scoring outputs. Named so a reader knows what the
    #: hashes do NOT vouch for.
    unhashed: List[str] = field(default_factory=list)
    #: The verified signature record, or None when the package is unsigned
    #: and a signature was not required.
    signature: Optional[Dict[str, Any]] = None
    #: Why the signature did not verify, when it did not.
    signature_error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.mismatched and not self.missing and self.signature_error is None

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["ok"] = self.ok
        return out


def _filename_for(entry: str) -> str:
    """A `content_hashes` key is a filename, except for the Parquet
    artifacts, which are recorded by name without their extension."""
    return entry if "." in entry else f"{entry}.parquet"


def verify_model_package(
    model_id: str,
    *,
    require_signature: bool = False,
    public_key: Union[bytes, str, None] = None,
) -> PackageVerification:
    """
    Hash every artifact the manifest covers and compare; check the
    signature when there is one or when one is required.
    """
    manifest = load_manifest(model_id)
    store = LocalArtifactStore()
    report = PackageVerification(model_id=model_id)
    present = {key.split("/", 1)[1] for key in store.list(model_id)}
    covered = set()
    for entry, expected in sorted(manifest.content_hashes.items()):
        filename = _filename_for(entry)
        covered.add(filename)
        key = f"{model_id}/{filename}"
        if not store.exists(key):
            report.missing.append(filename)
        elif store.hash(key) != expected:
            report.mismatched.append(filename)
        else:
            report.verified.append(filename)
    report.unhashed = sorted(present - covered - {MANIFEST_FILE})
    signed = SIGNATURE_FILE in present
    if signed or require_signature:
        try:
            report.signature = verify_manifest_signature(
                model_id, public_key=public_key
            )
        except ValidationError as exc:
            report.signature_error = str(exc)
    return report


def mirror_model_package(
    model_id: str,
    store: ArtifactStore,
    *,
    prefix: Optional[str] = None,
) -> Dict[str, str]:
    """
    Copy a verified package to `store`, re-hashing every covered file
    through the target; returns `{filename: uri}` on the target.

    A package that does not verify locally is refused rather than copied,
    because a mirror of a tampered package is a tampered package with a
    second address.
    """
    local = verify_model_package(model_id)
    if local.missing or local.mismatched:
        raise ValidationError(
            f"refusing to mirror model {model_id!r}: the local package does "
            f"not verify (missing {local.missing}, mismatched {local.mismatched})."
        )
    manifest = load_manifest(model_id)
    expected_by_filename = {
        _filename_for(entry): digest
        for entry, digest in manifest.content_hashes.items()
    }
    target_prefix = prefix or model_id
    validate_identifier(target_prefix, "prefix")
    source = LocalArtifactStore()
    filenames = [key.split("/", 1)[1] for key in source.list(model_id)]
    ordered = [f for f in filenames if f != MANIFEST_FILE] + [MANIFEST_FILE]
    uris: Dict[str, str] = {}
    for filename in ordered:
        data = source.get(f"{model_id}/{filename}")
        key = f"{target_prefix}/{filename}"
        uris[filename] = store.put(key, data)
        expected = expected_by_filename.get(filename)
        if expected is not None:
            landed = store.hash(key)
            if landed != expected:
                raise ValidationError(
                    f"{filename} did not survive the copy to {store.uri(key)}: "
                    f"registered {expected}, found {landed}."
                )
    return uris


__all__ = [
    "MANIFEST_FILE",
    "PROMOTIONS_FILE",
    "PackageVerification",
    "mirror_model_package",
    "verify_model_package",
]
