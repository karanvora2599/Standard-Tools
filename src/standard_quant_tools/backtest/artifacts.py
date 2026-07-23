"""
Local Parquet artifact store for backtest results too large to embed
inline in an agent-tool response (equity curves, trade logs) — the piece
BacktestResultV2 needs so it can report equity_curve_uri/trades_uri instead
of the full data, closing the "agent tool result can contain the complete
equity curve" gap noted for the plain BacktestResult. Same env-var-override
convention as SQT_AUDIT_DIR/SQT_CACHE_DIR.
"""

import os
import re
import uuid
from pathlib import Path
from typing import Union

import pandas as pd

from standard_quant_tools.error import ValidationError

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _runs_dir() -> Path:
    return Path(os.environ.get(
        "SQT_RUNS_DIR",
        str(Path.home() / ".cache" / "standard_quant_tools" / "runs"),
    ))


def _validate_identifier(value: str, field_name: str) -> None:
    """
    run_id/name are LLM-reachable (e.g. BacktestCompactInput.run_id) and get
    joined directly into a filesystem path — reject anything but a plain
    slug so path separators, '..', null bytes, or a drive-letter/absolute
    prefix can never make it into the path in the first place.
    """
    if not value or not _IDENTIFIER_RE.match(value):
        raise ValidationError(
            f"{field_name}={value!r} is not a valid identifier — only letters, "
            "digits, '_', and '-' are allowed (no path separators, '..', or empty string)."
        )


def _resolved_within_runs_dir(path: Path) -> Path:
    """Defense in depth on top of _validate_identifier: confirm the final
    resolved path is actually inside SQT_RUNS_DIR before any read/write."""
    root = _runs_dir().resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValidationError(f"resolved path {resolved} escapes SQT_RUNS_DIR ({root})")
    return resolved


def save_artifact(
    data: Union[pd.Series, pd.DataFrame], run_id: str, name: str, overwrite: bool = False,
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
    if isinstance(data, pd.Series):
        frame = data.to_frame(name=data.name or "value")
    else:
        frame = data
    if frame.empty:
        raise ValidationError("cannot save an empty artifact")

    directory = _runs_dir() / run_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.parquet"
    if path.exists() and not overwrite:
        raise ValidationError(
            f"artifact already exists at {path} (run_id={run_id!r}, name={name!r}) — "
            "pass overwrite=True to replace it, or use a different run_id/name."
        )
    frame.to_parquet(path)
    return str(path)


def load_artifact(uri: str) -> pd.DataFrame:
    """
    Read back an artifact saved by save_artifact. Always returns a
    DataFrame — if the original was a pd.Series, call
    `.squeeze("columns")` on the result to get an equivalent Series back
    (a no-op, returning the DataFrame unchanged, if there's more than one
    column).

    Raises:
        ValidationError: uri does not exist.
    """
    path = Path(uri)
    if not path.exists():
        raise ValidationError(f"artifact not found: {uri}")
    return pd.read_parquet(path)
