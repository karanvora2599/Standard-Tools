"""
Local Parquet artifact store for backtest results too large to embed
inline in an agent-tool response (equity curves, trade logs) — the piece
BacktestResultV2 needs so it can report equity_curve_uri/trades_uri instead
of the full data, closing the "agent tool result can contain the complete
equity curve" gap noted for the plain BacktestResult. Same env-var-override
convention as SQT_AUDIT_DIR/SQT_CACHE_DIR.
"""

import os
import uuid
from pathlib import Path
from typing import Union

import pandas as pd

from standard_quant_tools._runspath import (
    resolve_within_runs_dir as _resolved_within_runs_dir,
)
from standard_quant_tools._runspath import runs_dir as _runs_dir
from standard_quant_tools._runspath import validate_identifier as _validate_identifier
from standard_quant_tools.error import ValidationError

# These three were defined here and again in `modeling.artifacts`, identically
# and independently -- a path-traversal guard kept in two places, where
# hardening one silently leaves the other. They live in `_runspath` now. The
# private names are kept because they are this module's published surface:
# `agent.runtimes.handoff` imports all three from here.


def save_artifact(
    data: Union[pd.Series, pd.DataFrame],
    run_id: str,
    name: str,
    overwrite: bool = False,
) -> str:
    """
    Write data as Parquet under SQT_RUNS_DIR/<run_id>/<name>.parquet,
    returning the file path as a URI string. A pd.Series is converted to a
    single-column DataFrame first (named after the Series' own .name, or
    "value" if unnamed) — Parquet has no native Series concept; see
    load_artifact for how to get an equivalent Series back.

    Args:
        overwrite: run_id is caller-supplied (e.g. an agent-chosen or
            user-chosen id, not always a fresh uuid), so by default a
            second save_artifact call reusing the same (run_id, name) raises
            instead of silently clobbering the first run's artifact. Pass
            True to intentionally overwrite.

    Raises:
        ValidationError: data is empty, or the target file already exists
        and overwrite=False.
    """
    _validate_identifier(run_id, "run_id")
    _validate_identifier(name, "name")

    if isinstance(data, pd.Series):
        frame = data.to_frame(name=data.name or "value")
    else:
        frame = data
    if frame.empty:
        raise ValidationError("cannot save an empty artifact")

    directory = _runs_dir() / run_id
    path = directory / f"{name}.parquet"
    path = _resolved_within_runs_dir(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ValidationError(
            f"artifact already exists at {path} (run_id={run_id!r}, name={name!r}) — "
            "pass overwrite=True to replace it, or use a different run_id/name."
        )

    # Write atomically: a crash or concurrent reader must never observe a
    # partially-written Parquet file at the final path.
    tmp_path = directory / f".{name}.{uuid.uuid4().hex}.tmp"
    frame.to_parquet(tmp_path)
    os.replace(tmp_path, path)
    return str(path)


def load_artifact(uri: str) -> pd.DataFrame:
    """
    Read back an artifact saved by save_artifact. Always returns a
    DataFrame — if the original was a pd.Series, call
    `.squeeze("columns")` on the result to get an equivalent Series back
    (a no-op, returning the DataFrame unchanged, if there's more than one
    column).

    Raises:
        ValidationError: uri does not exist, or resolves outside SQT_RUNS_DIR.
    """
    path = _resolved_within_runs_dir(Path(uri))
    if not path.exists():
        raise ValidationError(f"artifact not found: {uri}")
    return pd.read_parquet(path)
