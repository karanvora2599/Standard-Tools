"""Content-fingerprint hashing shared by every other module in this package:
`hash_payload` for JSON-serializable objects (decision records, chain-index
entries), `hash_dataframe` for OHLCV data provenance."""

import hashlib
import json
from typing import Any


def hash_dataframe(df: Any) -> str:
    """Content fingerprint of a DataFrame (values + index), stable across runs."""
    import numpy as np
    import pandas as pd

    hashed = pd.util.hash_pandas_object(df, index=True)
    return hashlib.sha256(np.asarray(hashed.values).tobytes()).hexdigest()[:16]


def hash_payload(obj: Any) -> str:
    """Content fingerprint of a JSON-serializable object (dict/list/scalar)."""
    canonical = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
