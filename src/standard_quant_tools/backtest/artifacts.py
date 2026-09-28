"""
Local Parquet artifact store for backtest results too large to embed
inline in an agent-tool response (equity curves, trade logs) — the piece
BacktestResultV2 needs so it can report equity_curve_uri/trades_uri instead
of the full data, closing the "agent tool result can contain the complete
equity curve" gap noted for the plain BacktestResult. Same env-var-override
convention as SQT_AUDIT_DIR/SQT_CACHE_DIR.
"""

import io
from pathlib import Path
from typing import Union

import pandas as pd

from standard_quant_tools._runspath import (
    resolve_within_runs_dir as _resolved_within_runs_dir,
)
from standard_quant_tools._runspath import runs_dir as _runs_dir
from standard_quant_tools._runspath import validate_identifier as _validate_identifier
from standard_quant_tools.artifact_store import (
    write_bytes_atomically,
    write_bytes_exclusively,
)
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
            instead of silently clobbering the first run's artifact -- and
            of two calls racing to the same pair, exactly one succeeds.
            Pass True to intentionally overwrite.

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

    def _exists() -> ValidationError:
        return ValidationError(
            f"artifact already exists at {path} (run_id={run_id!r}, name={name!r}) — "
            "pass overwrite=True to replace it, or use a different run_id/name."
        )

    # A cheap early refusal before a large frame is serialised. It is NOT
    # the guard: two callers can both pass it, which is how two publishes
    # to one reference both succeeded and the last rename won.
    if path.exists() and not overwrite:
        raise _exists()

    # Written atomically either way: a crash or concurrent reader must never
    # observe a partially-written Parquet file at the final path. Without
    # overwrite the write is also EXCLUSIVE, and that is the guard -- of any
    # number of concurrent writers to one name exactly one succeeds, and the
    # rest are refused here with the message above.
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    if overwrite:
        write_bytes_atomically(path, buffer.getvalue())
    elif not write_bytes_exclusively(path, buffer.getvalue()):
        raise _exists()
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
