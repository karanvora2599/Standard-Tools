"""
The artifact store: one place where bytes touch storage.

WHY A PROTOCOL. `backtest.artifacts` and `modeling.artifacts` each wrote
their own files -- a temp name, the bytes, an `os.replace` -- and each
hashed them with its own loop, so the two properties every integrity
check in the model registry rests on (a write is atomic; a hash is of the
bytes on disk) were implemented twice and only equal by inspection. They
are implemented once here. The registry's package operations
(`modeling.registry.package`) are then written against the protocol, so
verifying a package and mirroring it to another store are the same code
whether the other store is a directory or a bucket.

WHAT IT IS NOT. `LocalArtifactStore` is the default and the only store
the runtime itself reads and writes: the registry still addresses its
directory by path -- listing models, checking a manifest exists,
appending to a promotion log -- and none of that goes through a store
yet. `FsspecArtifactStore` is therefore a TARGET, for mirroring a
verified package to object storage and reading it back, not a place the
runtime can be pointed at; saying otherwise would be a claim the code
does not make good on.

KEYS, NOT PATHS. A key is `<run_id>/<filename>`, relative to the store's
root. The run segment is the same slug `_runspath.validate_identifier`
accepts; a filename may carry dots but may not start with one, so `..`
and the store's own temp files are both unspellable. The local store
resolves every key inside its root before touching it, the same defence
in depth `_runspath` applies to paths.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import BinaryIO, List, Optional, Protocol, runtime_checkable

from standard_quant_tools._containment import require_within
from standard_quant_tools._runspath import runs_dir, validate_identifier
from standard_quant_tools.error import ValidationError

#: Digest length every content hash in the registry uses: SHA-256, the
#: first 16 hex characters -- the same length `audit/hashing.py` uses.
HASH_HEX_CHARS = 16
_CHUNK = 1024 * 1024
#: A filename inside a run: letters, digits, `_`, `-` and `.`, never
#: starting with a dot. That one rule excludes `..` and the `.name.tmp`
#: files an atomic write leaves behind if it is interrupted.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")


def validate_key(key: str) -> str:
    """`<run_id>/<filename>`, or a bare `<run_id>` for a prefix."""
    if not key or key.startswith("/") or "\\" in key or key.endswith("/"):
        raise ValidationError(
            f"artifact key {key!r} must be '<run_id>/<filename>' with forward "
            "slashes, no leading or trailing slash."
        )
    parts = key.split("/")
    if len(parts) > 2:
        raise ValidationError(
            f"artifact key {key!r} has {len(parts)} segments; a store is flat, "
            "one run directory deep, like SQT_RUNS_DIR itself."
        )
    validate_identifier(parts[0], "run_id")
    if len(parts) == 2 and not _FILENAME_RE.match(parts[1]):
        raise ValidationError(
            f"artifact filename {parts[1]!r} in key {key!r} may use letters, "
            "digits, '_', '-' and '.', and may not start with '.'."
        )
    return key


def hash_stream(handle: BinaryIO) -> str:
    """SHA-256 of everything left in `handle`, truncated to the registry's
    digest length."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(_CHUNK), b""):
        digest.update(chunk)
    return digest.hexdigest()[:HASH_HEX_CHARS]


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:HASH_HEX_CHARS]


def write_bytes_atomically(path: Path, data: bytes) -> None:
    """
    Write `data` so that no reader ever observes a partial file at `path`:
    the bytes go to a temp name in the same directory and are renamed
    over the target in one `os.replace`.

    The one implementation of this. It was written in `backtest.artifacts`
    for Parquet and again in `modeling.artifacts` for JSON and joblib, and
    the two were equal only until one of them was changed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp_path.write_bytes(data)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@runtime_checkable
class ArtifactStore(Protocol):
    """Bytes in, bytes out, by key; the hash is of what the store holds."""

    def put(self, key: str, data: bytes) -> str:
        """Store `data` at `key`, replacing what is there; return its URI."""
        ...

    def get(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def list(self, prefix: str = "") -> List[str]:
        """Every key under `prefix` (a run id, or empty for all), sorted."""
        ...

    def hash(self, key: str) -> str:
        """The registry digest of the bytes the store holds at `key`."""
        ...

    def uri(self, key: str) -> str: ...


class LocalArtifactStore:
    """
    A directory. `root` defaults to `SQT_RUNS_DIR`, resolved on every call
    rather than at construction so a store built at import time follows
    the environment the way `_runspath.runs_dir` does.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self._root = Path(root) if root is not None else None

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else runs_dir()

    def _path(self, key: str) -> Path:
        validate_key(key)
        root = self.root.resolve()
        resolved = (root / key).resolve()
        return require_within(
            resolved, root, f"artifact key {key!r} escapes the store root {root}"
        )

    def put(self, key: str, data: bytes) -> str:
        path = self._path(key)
        write_bytes_atomically(path, data)
        return str(path)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise ValidationError(f"artifact not found: {key}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str = "") -> List[str]:
        base = self._path(prefix) if prefix else self.root.resolve()
        if not base.is_dir():
            return []
        keys: List[str] = []
        root = self.root.resolve()
        for run in sorted(
            p for p in ([base] if prefix else base.iterdir()) if p.is_dir()
        ):
            for file in sorted(run.iterdir()):
                if file.is_file() and not file.name.startswith("."):
                    keys.append(file.relative_to(root).as_posix())
        return keys

    def hash(self, key: str) -> str:
        path = self._path(key)
        if not path.is_file():
            raise ValidationError(f"artifact not found: {key}")
        with open(path, "rb") as handle:
            return hash_stream(handle)

    def uri(self, key: str) -> str:
        return str(self._path(key))


def fsspec_available() -> bool:
    try:
        import fsspec  # noqa: F401
    except ImportError:
        return False
    return True


def require_fsspec() -> None:
    if not fsspec_available():
        raise ValidationError(
            "an object-store artifact target needs the `fsspec` package "
            "(`pip install standard_quant_tools[remote]`, plus the filesystem "
            "implementation for your store, e.g. `s3fs`)."
        )


class FsspecArtifactStore:
    """
    Any filesystem `fsspec` can open, addressed by URL: `s3://bucket/runs`,
    `gcs://...`, `memory://tests/runs`. A put is one object write, and
    whether that is atomic is the store's own property -- S3 and GCS
    make an object visible all at once; a networked POSIX mount may not.
    """

    def __init__(self, url: str, **storage_options) -> None:
        require_fsspec()
        import fsspec

        self._fs, root = fsspec.core.url_to_fs(url, **storage_options)
        self._root = str(root).rstrip("/")
        self._url = url

    def _path(self, key: str) -> str:
        validate_key(key)
        return f"{self._root}/{key}"

    def put(self, key: str, data: bytes) -> str:
        path = self._path(key)
        self._fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
        self._fs.pipe_file(path, data)
        return self.uri(key)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not self._fs.exists(path):
            raise ValidationError(f"artifact not found: {self.uri(key)}")
        return bytes(self._fs.cat_file(path))

    def exists(self, key: str) -> bool:
        path = self._path(key)
        return bool(self._fs.exists(path)) and bool(self._fs.isfile(path))

    def list(self, prefix: str = "") -> List[str]:
        base = self._path(prefix) if prefix else self._root
        if not self._fs.exists(base):
            return []
        keys = []
        for found in self._fs.find(base):
            relative = str(found)[len(self._root) + 1 :]
            name = relative.rsplit("/", 1)[-1]
            if relative and not name.startswith("."):
                keys.append(relative)
        return sorted(keys)

    def hash(self, key: str) -> str:
        path = self._path(key)
        if not self._fs.exists(path):
            raise ValidationError(f"artifact not found: {self.uri(key)}")
        with self._fs.open(path, "rb") as handle:
            return hash_stream(handle)

    def uri(self, key: str) -> str:
        return str(self._fs.unstrip_protocol(self._path(key)))


def store_from_url(url: str, **storage_options) -> ArtifactStore:
    """A local store for a bare path or `file://`, an fsspec store for
    anything with another scheme."""
    if "://" not in url:
        return LocalArtifactStore(Path(url))
    if url.startswith("file://"):
        return LocalArtifactStore(Path(url[len("file://") :]))
    return FsspecArtifactStore(url, **storage_options)


__all__ = [
    "HASH_HEX_CHARS",
    "ArtifactStore",
    "FsspecArtifactStore",
    "LocalArtifactStore",
    "fsspec_available",
    "hash_bytes",
    "hash_stream",
    "require_fsspec",
    "store_from_url",
    "validate_key",
    "write_bytes_atomically",
]
