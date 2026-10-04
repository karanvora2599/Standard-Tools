"""
A version-1 content hash reproduced under another pandas's representation.

Version 1 is `hashing.hash_dataframe`, which covers each column's
`str(dtype)` and the row hashes `pd.util.hash_pandas_object` computes from
the stored values. Both belong to the pandas version: pandas 3 prints a text
column as `str` where pandas 2 prints `object`, and parses a date string to
`datetime64[s]` (a millisecond epoch to `[ms]`, a date range to `[us]`)
where pandas 2 gives `datetime64[ns]`, and a datetime's row hash is taken
over its stored integers. The same data hashed under the two therefore gives
two values, and a check that compares one with the other refuses unchanged
data.

`legacy_hash_variant` re-hashes a frame as another pandas would have and
says which representation reproduces a recorded value. Two checks use it: a
dataset's `data_hash` (`modeling.dataset.integrity`) and a decision
record's `data_sources[].content_hash` (`replay`). Version 2,
`hashing.canonical_frame_hash`, needs none of this: it does not depend on
the pandas version.
"""

from __future__ import annotations

import hashlib
import json
import logging
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .hashing import hash_dataframe

logger = logging.getLogger(__name__)

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


def _at_unit(values: Any, unit: str, target: str) -> Optional[Any]:
    """`values` (a datetime column or a DatetimeIndex) at `target`, or None
    when converting there and back does not give the same instants."""
    import pandas as pd

    def convert(data: Any, to: str) -> Any:
        return data.dt.as_unit(to) if isinstance(data, pd.Series) else data.as_unit(to)

    try:
        converted = convert(values, target)
        lossless = convert(converted, unit).equals(values)
    except (OverflowError, ValueError):
        return None
    return converted if lossless else None


def legacy_hash_variant(
    frame: Any, expected: str, *, as_read: Optional[str] = None
) -> Optional[str]:
    """
    How `hash_dataframe` reproduces `expected` from this frame under
    another pandas's representation, described, or None when nothing
    tried does.

    Tried: each text column's dtype spelled `object`, `str` or `string`
    (per column, or every column alike past `_MAX_PER_COLUMN_TEXT` text
    columns), which changes only the hashed schema; then the datetime
    columns that share a resolution moved together to `[ns]`, `[us]`,
    `[ms]` or `[s]`, and a datetime index moved on its own (a provider
    builds its index with other code than its columns), where converting
    there and back gives the same instants, with each spelling again. Each
    column is hashed once, each datetime column and index once more per
    resolution, and a variant recombines the per-row hashes rather than
    re-hashing the frame -- so the whole search costs about one more
    `hash_dataframe` pass, and a respelling alone costs nothing past that.
    """
    import numpy as np
    import pandas as pd
    from pandas.api import types as pdt

    if frame.shape[1] == 0:
        return None
    columns = [frame.iloc[:, position] for position in range(frame.shape[1])]
    names = [str(name) for name in frame.columns]
    dtypes = [str(column.dtype) for column in columns]
    text = [position for position, column in enumerate(columns) if _is_text(column)]
    as_read = as_read if as_read is not None else hash_dataframe(frame)

    # Per-column row hashes, folded together the way `hash_pandas_object`
    # folds them. The fold is checked exactly: it has to give
    # `hash_dataframe`'s own value for the frame as read before any variant
    # built on it means anything.
    hashed = [
        pd.util.hash_pandas_object(column, index=False).to_numpy() for column in columns
    ]
    index_hash = pd.util.hash_pandas_object(frame.index, index=False).to_numpy()
    digest = _values_digest(hashed + [index_hash])
    recombined = _version_1(names, dtypes, digest) == as_read
    if not recombined:
        # A pandas that folds columns differently: its own fold for the
        # respellings, and no datetime variants, which need this one.
        rows = pd.util.hash_pandas_object(frame, index=True).to_numpy()
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

    # The members whose resolution is varied: the columns by position, and
    # the index after them. The index is not in the hashed schema, so
    # moving it changes only its row hashes.
    members: List[Any] = columns + [frame.index]
    index_position = len(columns)
    groups: List[Tuple[str, str, List[int]]] = []
    by_unit: Dict[str, List[int]] = {}
    for position, column in enumerate(columns):
        if pdt.is_datetime64_any_dtype(column.dtype):
            by_unit.setdefault(_datetime_unit(column), []).append(position)
    for unit, positions in by_unit.items():
        groups.append((f"[{unit}] datetimes", unit, positions))
    if isinstance(frame.index, pd.DatetimeIndex):
        unit = _datetime_unit(frame.index)
        groups.append((f"the [{unit}] index", unit, [index_position]))
    if not groups:
        return None

    # Per group, each lossless target: {position: (row hashes, dtype)}.
    options: List[List[Tuple[str, Dict[int, Tuple[Any, str]]]]] = []
    for _label, unit, positions in groups:
        choices: List[Tuple[str, Dict[int, Tuple[Any, str]]]] = []
        for target in _UNITS:
            if target == unit:
                choices.append((target, {}))
                continue
            moved: Dict[int, Tuple[Any, str]] = {}
            for position in positions:
                converted = _at_unit(members[position], unit, target)
                if converted is None:
                    break
                moved[position] = (
                    pd.util.hash_pandas_object(converted, index=False).to_numpy(),
                    str(converted.dtype),
                )
            else:
                choices.append((target, moved))
        options.append(choices)

    for combination in product(*options):
        if all(
            target == unit
            for (_label, unit, _), (target, _) in zip(groups, combination)
        ):
            continue  # the frame as read, tried above
        arrays = list(hashed) + [index_hash]
        kinds = list(dtypes)
        for _target, moved in combination:
            for position, (array, dtype) in moved.items():
                arrays[position] = array
                if position != index_position:
                    kinds[position] = dtype
        digest = _values_digest(arrays)
        found = _respelled(names, kinds, text, digest, expected)
        if found is not None:
            moves = [
                f"{label} at [{target}]"
                for (label, unit, _), (target, _moved) in zip(groups, combination)
                if target != unit
            ]
            return "; ".join(moves + ([] if found == "as read" else [found]))
    return None


__all__ = ["legacy_hash_variant"]
