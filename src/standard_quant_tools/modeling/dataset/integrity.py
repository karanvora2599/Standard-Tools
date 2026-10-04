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
"""

from __future__ import annotations

import logging
import platform
from typing import Any, Dict, Mapping, Optional

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
    that hashed it.
    """
    import pandas as pd

    stored = meta.get("data_hash")
    if stored is None:
        return
    version = int(meta.get("data_hash_version") or 1)
    external = meta.get("storage") == "external"
    rebuild = "Register the panel again." if external else "Rebuild the dataset."

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
                f"that does not describe the data actually used. {rebuild}"
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
            f"describe the data actually used. {rebuild}"
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
        "recorded it, and this check cannot tell which. "
        + ("Registering the panel again" if external else "Rebuilding the dataset")
        + " records a hash that does not depend on the pandas version."
    )


__all__ = [
    "DATA_HASH_VERSION",
    "build_environment",
    "legacy_hash_variant",
    "panel_data_hash",
    "verify_panel_hash",
]
