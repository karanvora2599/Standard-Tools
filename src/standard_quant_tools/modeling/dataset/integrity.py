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
refusal says so.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from itertools import product
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from standard_quant_tools.audit.hashing import canonical_frame_hash, hash_dataframe
from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

#: The form of `data_hash` a new dataset records. 1 is `hash_dataframe`
#: (metadata without the key), 2 is `canonical_frame_hash`.
DATA_HASH_VERSION = 2

#: How each pandas prints a text column's dtype: `object` (pandas 2), `str`
#: (pandas 3's default) and `string` (an explicit StringDtype under either).
_TEXT_SPELLINGS = ("object", "str", "string")

#: The datetime resolutions a version-1 hash is re-tried at, likeliest
#: first: pandas 2 parses a date string to `[ns]`, pandas 3 to `[s]`.
_UNITS = ("ns", "s", "us", "ms")

#: More text columns than this and the spellings are tried uniformly (every
#: text column spelled alike) rather than per column: 3**6 is 729 schema
#: hashes, each a few microseconds, and a panel has one or two.
_MAX_PER_COLUMN_TEXT = 6


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


# ── version 1: hash_dataframe under another pandas's representation ──────


def _is_text(values: Any) -> bool:
    import pandas as pd
    from pandas.api import types as pdt

    if isinstance(values.dtype, pd.StringDtype):
        return True
    if values.dtype == object:
        return pdt.infer_dtype(values, skipna=True) in ("string", "empty")
    return False


def _datetime_unit(values: Any) -> str:
    import numpy as np

    dtype = values.dtype
    # DatetimeTZDtype carries `.unit`; a numpy datetime64 dtype does not.
    return str(getattr(dtype, "unit", None) or np.datetime_data(dtype)[0])


def _combine(arrays: Sequence[Any]) -> Any:
    """`pandas.core.util.hashing.combine_hash_arrays`, which
    `hash_pandas_object` uses to fold per-column hashes into one per row.
    Reproduced so each column is hashed once and recombined per variant;
    `legacy_hash_variant` checks that it gives `hash_dataframe`'s own value
    for the frame as read before using it."""
    import numpy as np

    num_items = len(arrays)
    mult = np.uint64(1000003)
    out = np.zeros_like(arrays[0]) + np.uint64(0x345678)
    for i, array in enumerate(arrays):
        inverse_i = num_items - i
        out ^= array
        out *= mult
        mult += np.uint64(82520 + inverse_i + inverse_i)
    out += np.uint64(97531)
    return out


def _version_1(names: Sequence[str], dtypes: Sequence[str], values_digest: str) -> str:
    """`hash_dataframe`'s final step, from its parts."""
    schema = json.dumps([[n, d] for n, d in zip(names, dtypes)], sort_keys=False)
    combined = f"{schema}|{values_digest}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def _values_digest(arrays: Sequence[Any]) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(_combine(arrays)).tobytes()).hexdigest()


def _spellings(text: Sequence[int]) -> List[Tuple[str, ...]]:
    if len(text) <= _MAX_PER_COLUMN_TEXT:
        return list(product(_TEXT_SPELLINGS, repeat=len(text)))
    return [(spelling,) * len(text) for spelling in _TEXT_SPELLINGS]


def _respelled(
    names: Sequence[str],
    dtypes: Sequence[str],
    text: Sequence[int],
    digest: str,
    expected: str,
) -> Optional[str]:
    """The text spelling under which `names`/`dtypes` and a values digest
    give `expected`, described, or None. Only the schema is re-hashed."""
    kinds = list(dtypes)
    for spelling in _spellings(text):
        for position, spelled in zip(text, spelling):
            kinds[position] = spelled
        if _version_1(names, kinds, digest) == expected:
            respelled = sorted(
                {
                    spelled
                    for position, spelled in zip(text, spelling)
                    if spelled != dtypes[position]
                }
            )
            return f"text spelled {'/'.join(respelled)}" if respelled else "as read"
    return None


def legacy_hash_variant(
    panel: Any, expected: str, *, as_read: Optional[str] = None
) -> Optional[str]:
    """
    How `hash_dataframe` reproduces `expected` from this panel under
    another pandas's representation, described, or None when nothing
    tried does.

    Tried: each text column's dtype spelled `object`, `str` or `string`
    (per column, or every column alike past `_MAX_PER_COLUMN_TEXT` text
    columns), which changes only the hashed schema; then the datetime
    columns that share a resolution moved together to `[ns]`, `[us]`,
    `[ms]` or `[s]` where converting there and back gives the same
    instants, with each spelling again. Each column is hashed once, each
    datetime column once more per resolution, and a variant recombines the
    per-row hashes rather than re-hashing the frame -- so the whole search
    costs about one more `hash_dataframe` pass, and a respelling alone
    costs nothing past that.
    """
    import numpy as np
    import pandas as pd
    from pandas.api import types as pdt

    if panel.shape[1] == 0:
        return None
    columns = [panel.iloc[:, position] for position in range(panel.shape[1])]
    names = [str(name) for name in panel.columns]
    dtypes = [str(column.dtype) for column in columns]
    text = [position for position, column in enumerate(columns) if _is_text(column)]
    as_read = as_read if as_read is not None else hash_dataframe(panel)

    # Per-column row hashes, folded together the way `hash_pandas_object`
    # folds them. The fold is checked exactly: it has to give
    # `hash_dataframe`'s own value for the frame as read before any variant
    # built on it means anything.
    hashed = [
        pd.util.hash_pandas_object(column, index=False).to_numpy() for column in columns
    ]
    index_hash = pd.util.hash_pandas_object(panel.index, index=False).to_numpy()
    digest = _values_digest(hashed + [index_hash])
    recombined = _version_1(names, dtypes, digest) == as_read
    if not recombined:
        # A pandas that folds columns differently: its own fold for the
        # respellings, and no datetime variants, which need this one.
        rows = pd.util.hash_pandas_object(panel, index=True).to_numpy()
        digest = hashlib.sha256(np.asarray(rows).tobytes()).hexdigest()
        if _version_1(names, dtypes, digest) != as_read:
            logger.debug(
                "version-1 hash variants not tried: hash_dataframe under pandas "
                "%s is not reproduced from its parts",
                pd.__version__,
            )
            return None

    # The values as read, the schema respelled.
    found = _respelled(names, dtypes, text, digest, expected)
    if found is not None:
        return found
    if not recombined:
        logger.debug(
            "version-1 datetime variants not tried: pandas %s folds column "
            "hashes differently from this module",
            pd.__version__,
        )
        return None

    groups: Dict[str, List[int]] = {}
    for position, column in enumerate(columns):
        if pdt.is_datetime64_any_dtype(column.dtype):
            groups.setdefault(_datetime_unit(column), []).append(position)
    if not groups:
        return None

    # Per resolution group, each lossless target: {position: (hash, dtype)}.
    options: List[List[Tuple[str, Dict[int, Tuple[Any, str]]]]] = []
    for unit, positions in groups.items():
        choices: List[Tuple[str, Dict[int, Tuple[Any, str]]]] = []
        for target in _UNITS:
            if target == unit:
                choices.append((target, {}))
                continue
            moved: Dict[int, Tuple[Any, str]] = {}
            for position in positions:
                original = columns[position]
                try:
                    converted = original.dt.as_unit(target)
                    lossless = converted.dt.as_unit(unit).equals(original)
                except (OverflowError, ValueError):
                    lossless = False
                if not lossless:
                    break
                moved[position] = (
                    pd.util.hash_pandas_object(converted, index=False).to_numpy(),
                    str(converted.dtype),
                )
            else:
                choices.append((target, moved))
        options.append(choices)

    group_units = list(groups)
    for combination in product(*options):
        if all(target == unit for unit, (target, _) in zip(group_units, combination)):
            continue  # the frame as read, tried above
        arrays = list(hashed)
        kinds = list(dtypes)
        for _target, moved in combination:
            for position, (array, dtype) in moved.items():
                arrays[position] = array
                kinds[position] = dtype
        digest = _values_digest(arrays + [index_hash])
        found = _respelled(names, kinds, text, digest, expected)
        if found is not None:
            moves = [
                f"[{unit}] datetimes at [{target}]"
                for unit, (target, _moved) in zip(group_units, combination)
                if target != unit
            ]
            return "; ".join(moves + ([] if found == "as read" else [found]))
    return None


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
