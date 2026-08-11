"""
sanitize_for_json — the single JSON-safety boundary shared by both agent
tool surfaces (agent.tools.dispatch and modeling.agent.modeling_dispatch).

A legitimately non-finite metric is valid math but not valid JSON per
RFC 8259: `sortino_ratio` when there is no downside deviation,
`profit_factor` with no losing trades, a classification AUC on a
single-class fold, or HistGradientBoosting's feature importance (which the
estimator simply does not expose). Python's `json.dumps` emits the
non-standard tokens `Infinity`/`-Infinity`/`NaN` for these, which many
strict parsers — including some LLM API backends — reject outright.

`None` is the JSON-safe way to say "this metric is undefined or unbounded"
without substituting a made-up finite number that would misrepresent the
result.

Lives at the top level rather than inside either agent package because
both need it and neither should import the other: the modeling runtime is
deliberately independent of the 46-tool surface.
"""

import math
import numbers
from typing import Any


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace non-finite floats with None."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    # Tuples/sets encode as JSON arrays but were not walked by the original
    # dict/list-only version, so a non-finite value inside one survived to
    # the encoder.
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [sanitize_for_json(v) for v in obj]
    if hasattr(obj, "tolist") and callable(obj.tolist):  # numpy array/scalar
        return sanitize_for_json(obj.tolist())
    # numbers.Real covers np.float32 as well as np.float64 (which happens to
    # subclass float) and plain floats.
    if isinstance(obj, numbers.Real) and not isinstance(obj, bool):
        return obj if math.isfinite(float(obj)) else None
    return obj
