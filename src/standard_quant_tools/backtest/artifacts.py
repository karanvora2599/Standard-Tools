"""
Local Parquet artifact store for backtest results too large to embed
inline in an agent-tool response (equity curves, trade logs) — the piece
BacktestResultV2 needs so it can report equity_curve_uri/trades_uri instead
of the full data, closing the "agent tool result can contain the complete
equity curve" gap noted for the plain BacktestResult. Same env-var-override
convention as SQT_AUDIT_DIR/SQT_CACHE_DIR.
"""

import os
from pathlib import Path
from typing import Union

import pandas as pd

from standard_quant_tools.error import ValidationError


def _runs_dir() -> Path:
    return Path(os.environ.get(
        "SQT_RUNS_DIR",
        str(Path.home() / ".cache" / "standard_quant_tools" / "runs"),
    ))


def save_artifact(data: Union[pd.Series, pd.DataFrame], run_id: str, name: str) -> str:
    """
    Write data as Parquet under SQT_RUNS_DIR/<run_id>/<name>.parquet,
    returning the file path as a URI string. A pd.Series is converted to a
    single-column DataFrame first (named after the Series' own .name, or
    "value" if unnamed) — Parquet has no native Series concept; see
    load_artifact for how to get an equivalent Series back.

    Raises:
        ValidationError: data is empty.
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
