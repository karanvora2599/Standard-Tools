"""
A dataset's content hash: what a build records, and how a load checks it.

Every dataset records `data_hash` when it is built or registered, and every
tool that reads the panel recomputes it and refuses a mismatch, because a
model trained on an edited panel would carry a lineage hash describing data
it never saw.

TWO FORMS. Version 1 (a dataset whose metadata has no `data_hash_version`)
is `audit.hash_dataframe`, which covers each column's `str(dtype)`. That
spelling belongs to the pandas version: pandas 3 prints a text column as
`str` where pandas 2 prints `object`, and parses a date string to
`datetime64[s]` where pandas 2 gives `datetime64[ns]`. A dataset built
under one was refused under the other as edited, though no byte of it had
changed. Version 2 is `audit.canonical_frame_hash`, which covers the same
names, types and values through a representation that is the same under
every pandas. New datasets record version 2, with the library versions that
built them under `built_with`.

NO MIGRATION. An existing dataset's `data_hash` is never rewritten: it is
copied into every model manifest trained from it and into each fold's node
hash, so rewriting it would change the recorded identity of runs that
already exist. A version-1 hash is verified as it was, and on a miss the
panel is re-hashed under the other pandas's representation -- text spelled
`object`, `str` or `string`, datetimes at `[ns]`, `[us]`, `[ms]` or `[s]`
where the conversion loses nothing -- and accepted when one of those
reproduces the recorded value, as is a version-2 value written into
metadata without its version key. SHA-256 makes a false match from a
changed value impossible; what the variants cannot do is reproduce a hash
recorded by a pandas whose differences are not in that list, and the
refusal says so. The variant search is `audit.legacy_hash`, which a replay
of a decision record recorded before its data sources were versioned uses
too.

AN EXTERNAL PANEL'S FILE. A panel registered by reference is re-read from
the caller's file on every load, and the registration records the file's
name, size and modification time: `panel_fingerprint`, their digest, and
`panel_file_stats`, which keeps them apart. A refusal of such a panel says
whether they still match. Writing to a file moves its modification time
unless something sets it back, so an unchanged file points to a difference
in how it was read or hashed, and a changed one to an edit.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import platform
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from standard_quant_tools.audit.hashing import canonical_frame_hash, hash_dataframe
from standard_quant_tools.audit.legacy_hash import legacy_hash_variant
from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

#: The form of `data_hash` a new dataset records. 1 is `hash_dataframe`
#: (metadata without the key), 2 is `canonical_frame_hash`.
DATA_HASH_VERSION = 2


def panel_data_hash(panel: Any) -> str:
    """The `data_hash` a new dataset records: version 2."""
    return canonical_frame_hash(panel)


def build_environment() -> Dict[str, Optional[str]]:
    """The library versions a dataset is built under, recorded as
    `built_with`. A version-1 hash depends on them, and a refusal of one
    names them."""
    import numpy as np
    import pandas as pd

    try:
        from importlib.metadata import version

        pyarrow_version: Optional[str] = version("pyarrow")
    except Exception:  # not installed: nothing to record
        pyarrow_version = None
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": pyarrow_version,
    }


# ── an external panel's file, as registered ─────────────────────────────


def panel_file_stats(path: Any) -> Dict[str, Any]:
    """
    The names, sizes and modification times of the files behind an
    external panel, recorded at registration as `panel_file_stats`.

    `panel_fingerprint` is one digest of the same three, so it can say that
    something moved and not what. This keeps them apart: `names`, `sizes`
    and `mtimes` are digests of each in file-name order (relative names in
    `/` form, so a directory hashes alike on every OS), beside the file
    count, the total size and the newest modification time in nanoseconds.
    The files are the ones `panel_fingerprint` covers. Each is stat'ed, and
    none is read.
    """
    from standard_quant_tools.data import external as _external

    root = Path(path)
    base = root if root.is_dir() else root.parent
    names, sizes, mtimes = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    files = total = 0
    newest: Optional[int] = None
    for file in _external._files(root):
        try:
            stat = file.stat()
        except OSError:  # pragma: no cover - raced deletion
            continue
        for digest, part in (
            (names, file.relative_to(base).as_posix()),
            (sizes, str(stat.st_size)),
            (mtimes, str(stat.st_mtime_ns)),
        ):
            encoded = part.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        files += 1
        total += int(stat.st_size)
        newest = (
            int(stat.st_mtime_ns) if newest is None else max(newest, stat.st_mtime_ns)
        )
    return {
        "files": files,
        "bytes": total,
        "modified_ns": newest,
        "names": names.hexdigest()[:16],
        "sizes": sizes.hexdigest()[:16],
        "mtimes": mtimes.hexdigest()[:16],
    }


def _count(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else "an unrecorded number of"


def _utc(ns: Any) -> str:
    """A modification time in nanoseconds, as UTC to the nanosecond."""
    if not isinstance(ns, int):
        return "an unrecorded time"
    seconds, fraction = divmod(ns, 10**9)
    moment = datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)
    return f"{moment:%Y-%m-%d %H:%M:%S}.{fraction:09d} UTC"


def panel_file_state(
    meta: Mapping[str, Any], panel_path: Any
) -> Optional[Tuple[bool, str]]:
    """
    Whether an external panel's file still has the name, size and
    modification time its registration recorded: (unchanged, a sentence
    saying so or naming what moved). None when the registration recorded
    neither `panel_file_stats` nor `panel_fingerprint` -- a built dataset,
    whose panel this library wrote -- or the file cannot be stat'ed.

    A registration made before `panel_file_stats` existed is compared by
    its fingerprint alone, which says whether anything moved but not what.
    """
    path = Path(str(panel_path))
    recorded = meta.get("panel_file_stats")
    fingerprint = meta.get("panel_fingerprint")
    try:
        directory = path.is_dir()
        if isinstance(recorded, Mapping):
            now = panel_file_stats(path)
        elif fingerprint:
            from standard_quant_tools.data import external as _external

            same = _external.fingerprint(path) == fingerprint
        else:
            return None
    except OSError:
        return None

    if directory:
        whose, what, also = "files'", "names, sizes and modification times", "their"
    else:
        whose, what, also = "file's", "name, size and modification time", "its"
    unchanged = f"The {whose} {what} are unchanged since registration."
    if not isinstance(recorded, Mapping):
        if same:
            return True, unchanged
        return False, (
            f"The {whose} {what.replace(' and ', ' or ')} changed since "
            "registration; the registration recorded their digest alone, so "
            "which of them is not known."
        )

    if now["names"] != recorded.get("names"):
        return False, (
            "Since registration, files were added to, removed from or renamed "
            f"in {path} ({_count(recorded.get('files'))} files at registration, "
            f"{_count(now['files'])} now)."
        )
    moved: List[str] = []
    if now["sizes"] != recorded.get("sizes"):
        then, size_now = _count(recorded.get("bytes")), _count(now["bytes"])
        moved.append(
            f"sizes changed ({then} bytes in all at registration, {size_now} now)"
            if directory
            else f"size changed from {then} to {size_now} bytes"
        )
    if now["mtimes"] != recorded.get("mtimes"):
        then, time_now = _utc(recorded.get("modified_ns")), _utc(now["modified_ns"])
        moved.append(
            f"modification times changed (the newest {then} at registration, "
            f"{time_now} now)"
            if directory
            else f"modification time changed from {then} to {time_now}"
        )
    if not moved:
        return True, unchanged
    return False, f"Since registration, the {whose} {f' and {also} '.join(moved)}."


#: Why an unchanged name, size and modification time is a hint at all.
_WRITES_MOVE_MTIME = (
    "Writing to a file moves its modification time unless something sets it back, so"
)


def _file_note(meta: Mapping[str, Any], panel_path: Any, *, undecided: bool) -> str:
    """For an external panel's refusal: whether its file moved since
    registration, and what that points to. Empty for a built dataset, and
    when the file cannot be examined: a hint never replaces the refusal."""
    try:
        state = panel_file_state(meta, panel_path)
    except Exception:  # noqa: BLE001 - the refusal stands without the hint
        logger.debug("panel file state not determined", exc_info=True)
        return ""
    if state is None:
        return ""
    unchanged, sentence = state
    if unchanged and undecided:
        return (
            f" {sentence} {_WRITES_MOVE_MTIME} a pandas difference is the "
            "likelier cause."
        )
    if unchanged:
        return (
            f" {sentence} {_WRITES_MOVE_MTIME} either an edit set it back or the "
            "file now parses to different values, as another version of its "
            "reader can."
        )
    if undecided:
        return f" {sentence} That points to an edit rather than a pandas difference."
    return f" {sentence}"


# ── the check every load runs ────────────────────────────────────────────


def _parquet_pandas_version(path: Any) -> Optional[str]:
    """The pandas version pyarrow recorded in a Parquet file's footer when
    pandas wrote it, or None. Read only on the refusal path."""
    try:
        import pyarrow.parquet as pq

        metadata = pq.read_schema(str(path)).pandas_metadata or {}
    except Exception:
        return None
    found = metadata.get("pandas_version")
    return str(found) if found else None


def verify_panel_hash(
    panel: Any,
    meta: Mapping[str, Any],
    *,
    dataset_id: str,
    panel_path: Any,
    written_parquet: Any = None,
) -> None:
    """
    Refuse a panel that does not match the `data_hash` its dataset
    recorded.

    `panel_path` is the file the refusal names. `written_parquet` is the
    Parquet file this library wrote for a built dataset, whose footer
    records the pandas that wrote it; None for an external panel, whose
    file was written by someone else and says nothing about the pandas
    that hashed it. An external panel's refusal also says whether its file
    has the name, size and modification time recorded at registration
    (`panel_file_state`).
    """
    import pandas as pd

    stored = meta.get("data_hash")
    if stored is None:
        return
    version = int(meta.get("data_hash_version") or 1)
    external = meta.get("storage") == "external"
    rebuild = "Register the panel again." if external else "Rebuild the dataset."

    def file_note(undecided: bool) -> str:
        # Read only on the refusal path: a stat of each file behind it.
        return _file_note(meta, panel_path, undecided=undecided) if external else ""

    if version == 2:
        actual = canonical_frame_hash(panel)
        if actual != stored:
            raise ValidationError(
                f"dataset {dataset_id!r}: the panel at {panel_path} no longer "
                "matches the content hash recorded when the dataset was built "
                f"(expected {stored}, found {actual}). This hash covers the "
                "column names, their types and every value, and does not "
                "depend on the pandas or pyarrow version, so the data has "
                "changed since the build. Using it would record a lineage hash "
                "that does not describe the data actually used."
                f"{file_note(False)} {rebuild}"
            )
        return
    if version != 1:
        raise ValidationError(
            f"dataset {dataset_id!r}: its content hash was recorded as version "
            f"{version}, and this release knows versions 1 and "
            f"{DATA_HASH_VERSION}. A later release of standard_quant_tools "
            "built it; load it with that release."
        )

    actual = hash_dataframe(panel)
    if actual == stored:
        return
    how = legacy_hash_variant(panel, stored, as_read=actual)
    if how is None and canonical_frame_hash(panel) == stored:
        # Metadata written by hand from a build's `data_hash` without its
        # version key. The value is a full content hash either way.
        how = "the version-2 hash, recorded without its version key"
    if how is not None:
        logger.debug(
            "dataset %s: version-1 hash %s reproduced with %s (pandas %s)",
            dataset_id,
            stored,
            how,
            pd.__version__,
        )
        return

    running = pd.__version__
    built_with = meta.get("built_with") or {}
    writer = built_with.get("pandas") if isinstance(built_with, Mapping) else None
    if writer is None and written_parquet is not None:
        writer = _parquet_pandas_version(written_parquet)
    if writer is not None and str(writer) == running:
        raise ValidationError(
            f"dataset {dataset_id!r}: the panel at {panel_path} no longer "
            "matches the hash recorded when the dataset was built (expected "
            f"{stored}, found {actual}). The panel was written by pandas "
            f"{writer} and this process runs pandas {running}, so the data has "
            "changed. Using it would record a lineage hash that does not "
            f"describe the data actually used.{file_note(False)} {rebuild}"
        )
    who = (
        f"The panel was written by pandas {writer} and this process runs "
        f"pandas {running}."
        if writer is not None
        else "The dataset does not record which pandas built it, and this "
        f"process runs pandas {running}."
    )
    raise ValidationError(
        f"dataset {dataset_id!r}: the panel at {panel_path} could not be "
        f"verified. Its hash ({stored}) was recorded by the earlier form of "
        "this check, which depends on how pandas names and stores dtypes. "
        f"{who} Re-hashing it under both versions' dtype names and datetime "
        f"resolutions does not reproduce {stored}, so either its values have "
        "changed or this pandas hashes them differently from the one that "
        f"recorded it, and this check cannot tell which.{file_note(True)} "
        + ("Registering the panel again" if external else "Rebuilding the dataset")
        + " records a hash that does not depend on the pandas version."
    )


__all__ = [
    "DATA_HASH_VERSION",
    "build_environment",
    "legacy_hash_variant",
    "panel_data_hash",
    "panel_file_state",
    "panel_file_stats",
    "verify_panel_hash",
]
