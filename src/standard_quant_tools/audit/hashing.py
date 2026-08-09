"""Content-fingerprint hashing shared by every other module in this package:
`hash_payload` for JSON-serializable objects (decision records, chain-index
entries), `hash_dataframe` for OHLCV data provenance."""

import hashlib
import json
from typing import Any


def hash_dataframe(df: Any) -> str:
    """
    Content fingerprint of a DataFrame (columns + values + index), stable
    across runs.

    Covers the COLUMN NAMES and dtypes as well as the values. Hashing values
    alone (pd.util.hash_pandas_object is a per-row digest that never sees the
    column labels) meant two frames holding identical numbers under entirely
    different column names produced the same fingerprint -- e.g. a
    Close/Open frame and a Volume/Adj frame collided, which defeats the point
    of a provenance hash whose whole job is to tell different data apart.

    NOTE (format change): fingerprints produced here differ from those written
    by versions before this fix. Replaying a decision record captured by an
    older version will report a data_source mismatch even when the underlying
    data is unchanged. Only the `content_hash` values inside `data_sources`
    are affected -- the tamper-evident record chain is built by `hash_payload`
    (below), which is unchanged for all normal records, so existing audit
    trails still verify.
    """
    import numpy as np
    import pandas as pd

    hashed = pd.util.hash_pandas_object(df, index=True)
    values_digest = hashlib.sha256(np.asarray(hashed.values).tobytes()).hexdigest()
    # Column identity, in the frame's own column order (a reordering is a
    # genuinely different frame for provenance purposes).
    schema = json.dumps(
        [[str(c), str(dtype)] for c, dtype in zip(df.columns, df.dtypes)]
        if hasattr(df, "columns")
        else [],
        sort_keys=False,
    )
    combined = f"{schema}|{values_digest}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def _canonical_default(obj: Any) -> Any:
    """
    Fallback encoder for objects `json` can't serialize natively.

    Plain `default=str` silently routed numpy arrays (and anything else with
    an abbreviating __str__) through a LOSSY repr: numpy truncates with '...'
    past ~1000 elements, so two different large arrays produced byte-identical
    canonical forms and therefore the same hash. Anything array-like is
    converted to its full element list here instead; genuinely opaque objects
    still fall back to str(), which is fine for the scalars (datetime, Path,
    Decimal) that actually reach this path in practice.
    """
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):  # numpy ndarray / scalar, pandas Series/Index
        return tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    return str(obj)


def hash_payload(obj: Any) -> str:
    """
    Content fingerprint of a JSON-serializable object (dict/list/scalar).

    Output is unchanged for objects made only of native JSON types (which is
    every DecisionRecord / chain-index entry), so the tamper-evident record
    chain built on this function stays valid across this change -- only the
    previously-lossy non-JSON fallback path behaves differently.
    """
    canonical = json.dumps(obj, sort_keys=True, default=_canonical_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
