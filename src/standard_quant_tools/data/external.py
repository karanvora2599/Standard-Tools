"""
Datasets too large to copy: registered where they lie, read in batches.

WHY THIS IS NOT A PROVIDER. Every other path into this library goes
`DataProvider.get_*` -> a whole `pd.DataFrame` -> `save_artifact` -> a second
whole copy under `SQT_RUNS_DIR`. That is two full materializations of the
same bytes, and for a day of L2 depth it is two materializations of
something that does not fit in memory once. The only concession to size
anywhere else in the surface is `fetch_tick_tape`'s `limit`, which does not
sample -- it TRUNCATES, so every rate and total computed downstream
understates the real one and nothing in the numbers says so.

So this module does the one thing the fetch path cannot: it takes data the
caller already has, on their own disk, and makes it addressable WITHOUT
moving it. What gets stored is a pointer and a schema. What gets read is a
batch at a time.

WHY PARQUET AND CSV ONLY. Not a limitation -- a containment boundary. The
path is caller-supplied and reaches this library from an agent, so "read the
file at this path and put it in a tool result" is a capability worth
bounding. A columnar or delimited reader that fails on anything else refuses
to be a general file-exfiltration primitive, and both formats cover what
market-data vendors actually ship.

WHAT A KIND IS FOR. The same thing it is for in `handoff.py`: a mismatched
handoff should fail by name, immediately, rather than several frames deep in
pandas. A registered dataset declares what it holds, and this module checks
the columns that claim implies -- `order_book_panel` without `ask_size_0` is
refused at registration rather than discovered by `book_metrics` returning a
null microprice for every snapshot.

WHAT REGISTRATION DOES NOT PROMISE. That the file will still be there, or
still be the same bytes, when someone resolves the reference. A published
artifact is immutable because this library wrote it; an external file
belongs to the caller and can change underneath. `fingerprint()` is what
makes that detectable rather than silent -- it is deliberately NOT a content
hash, because hashing forty gigabytes to answer "did this change" costs more
than the read it was meant to protect.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from standard_quant_tools.error import ValidationError

#: Suffix -> the pyarrow dataset format that reads it. A directory is
#: probed by what is inside it, so a partitioned Parquet dataset and a
#: single file take the same path through here.
_FORMATS: Dict[str, str] = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
    ".txt": "csv",
    ".tsv": "csv",
}

FORMATS: Tuple[str, ...] = ("parquet", "csv")

#: The columns each kind's consumers already require, named here so a bad
#: extract is refused at registration.
#:
#: These are not invented for this module. `tick_tape` and `quote_panel`
#: repeat the exact contract `handoff.KINDS` states -- "those exact names,
#: because the microstructure tools refuse without them" -- and
#: `order_book_panel` repeats `DataProvider.get_order_book`'s declared
#: columns, which `analysis/order_book.py` has read since before any source
#: existed to feed it. `event_panel` repeats `point_in_time.py`'s temporal
#: contract.
KIND_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "order_book_panel": (
        "timestamp",
        "bid_price_0",
        "bid_size_0",
        "ask_price_0",
        "ask_size_0",
    ),
    # Exactly `analysis.order_events.ORDER_EVENT_COLUMNS`. `price` was
    # missing here, so a panel could satisfy REGISTRATION and then fail
    # inside `order_event_metrics`, which requires all six.
    "order_event_panel": (
        "timestamp",
        "order_id",
        "action",
        "side",
        "price",
        "size",
    ),
    "event_panel": ("event_time", "available_time"),
    "tick_tape": ("price", "size"),
    "quote_panel": ("bid_price", "ask_price"),
}

KIND_DESCRIPTIONS: Dict[str, str] = {
    "order_book_panel": (
        "L2 depth snapshots: `timestamp`, then `bid_price_{i}` / "
        "`bid_size_{i}` / `ask_price_{i}` / `ask_size_{i}` for each level, "
        "level 0 being the touch. The shape `get_order_book_metrics` reads."
    ),
    "order_event_panel": (
        "Order-by-order events: `timestamp`, `order_id`, `action` (A/C/M/"
        "F/T/R), `side`, `price`, `size`. `order_id` and `action` are what "
        "make it an order feed rather than a book."
    ),
    "event_panel": (
        "Rows in event time carrying the point-in-time contract: "
        "`event_time` (when it describes the world) and `available_time` "
        "(when it could first be acted on, and what a join must use)."
    ),
    "tick_tape": "Individual trades with `price` and `size` columns.",
    "quote_panel": "Top-of-book quotes with `bid_price` and `ask_price`.",
}

#: Read this many rows at a time. Large enough that per-batch overhead is
#: noise, small enough that one batch of a wide depth book stays well under
#: a hundred megabytes.
DEFAULT_BATCH_ROWS = 65_536

#: How many rows validation reads before it stops and says so. A full scan
#: of a multi-billion-row tape is not a thing to do inside a tool call, and
#: a verdict that never returns is worth less than a bounded one that
#: reports what it covered.
DEFAULT_SCAN_LIMIT = 2_000_000


def _pyarrow_dataset():
    try:
        import pyarrow.dataset as arrow_dataset
    except ImportError as exc:  # pragma: no cover - pyarrow is a core dep
        raise ValidationError(
            "reading an external dataset needs pyarrow, which is a declared "
            "dependency of this library but is not importable here. Install "
            "it with `pip install pyarrow>=12`."
        ) from exc
    return arrow_dataset


def _infer_format(path: Path) -> str:
    """What reads this path, from its suffix or from what a directory holds."""
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                fmt = _FORMATS.get(_suffix(child))
                if fmt is not None:
                    return fmt
        raise ValidationError(
            f"{path} is a directory with no Parquet or CSV file in it. A "
            "directory is read as one partitioned dataset, so it needs at "
            f"least one file whose suffix is one of {sorted(_FORMATS)}."
        )
    suffix = _suffix(path)
    fmt = _FORMATS.get(suffix)
    if fmt is None:
        raise ValidationError(
            f"{path.name} has suffix {suffix or '(none)'}, which this "
            f"library does not read. Supported: {sorted(_FORMATS)}. The "
            "restriction is deliberate -- the path is caller-supplied, and a "
            "reader that accepts anything is a way to read any file on this "
            "machine into a tool result."
        )
    return fmt


def _suffix(path: Path) -> str:
    """The format-bearing suffix, seeing through one compression suffix."""
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] in (".gz", ".bz2", ".zst", ".lz4"):
        suffixes = suffixes[:-1]
    return suffixes[-1] if suffixes else ""


def resolve_path(path: str) -> Path:
    """
    Turn a caller-supplied path into one that exists, or say why not.

    Deliberately NOT constrained to `SQT_RUNS_DIR`, unlike everything in
    `backtest/artifacts.py`. That containment exists because run_id and name
    are agent-chosen slugs joined into a path; here the whole point is to
    reach data this library did not write and will not copy. The bound that
    replaces it is format: only Parquet and CSV are readable, so a path
    that is not tabular market data refuses.
    """
    text = str(path).strip()
    if not text:
        raise ValidationError(
            "an external dataset needs a path; got an empty string. Pass the "
            "file, or the directory holding a partitioned dataset."
        )
    resolved = Path(os.path.expandvars(os.path.expanduser(text))).resolve()
    if not resolved.exists():
        raise ValidationError(
            f"no file or directory at {resolved}. Nothing is copied when a "
            "dataset is registered, so the path has to be readable from "
            "wherever this library runs, not only from where it was typed."
        )
    return resolved


def _files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file())


def fingerprint(path: Path) -> str:
    """
    A cheap, honest answer to "has this changed since it was registered".

    NOT a content hash, and named so it cannot be mistaken for one. It is
    the digest of every file's relative name, byte size and modification
    time. That catches a re-extract, a truncated copy and a partially
    written file; it does not catch an edit that preserves both size and
    mtime. Hashing the bytes would catch that too and would cost a full read
    of data this module exists to avoid fully reading -- so the weaker check
    that always runs beats the stronger one nobody would wait for.
    """
    digest = hashlib.sha256()
    root = path if path.is_dir() else path.parent
    for file in _files(path):
        try:
            stat = file.stat()
        except OSError:  # pragma: no cover - raced deletion
            continue
        digest.update(str(file.relative_to(root)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def total_bytes(path: Path) -> int:
    total = 0
    for file in _files(path):
        try:
            total += file.stat().st_size
        except OSError:  # pragma: no cover - raced deletion
            continue
    return total


@dataclass(frozen=True)
class ExternalDataset:
    """
    A handle to data that stays where it is.

    What `resolve()` hands back for an externally-registered reference,
    instead of a DataFrame. That difference is the entire point and is why
    this is a distinct type rather than a lazily-loaded frame: a caller has
    to decide, in code, how much of it to bring into memory. Something
    shaped like a DataFrame would let a consumer written for a fetched panel
    silently pull forty gigabytes through an `.iloc`.
    """

    path: Path
    kind: str
    fmt: str
    columns: Tuple[str, ...]
    dtypes: Dict[str, str]
    rows: Optional[int] = None
    n_files: int = 0
    size_bytes: int = 0
    fingerprint: str = ""

    def scanner(self, *, columns: Optional[Sequence[str]] = None, batch_rows: int = 0):
        dataset = open_dataset(self.path, fmt=self.fmt)
        selected = self._checked_columns(columns)
        return dataset.scanner(
            columns=selected,
            batch_size=int(batch_rows) if batch_rows else DEFAULT_BATCH_ROWS,
        )

    def _checked_columns(self, columns: Optional[Sequence[str]]) -> Optional[List[str]]:
        if columns is None:
            return None
        wanted = [str(c) for c in columns]
        unknown = [c for c in wanted if c not in self.columns]
        if unknown:
            raise ValidationError(
                f"{self.path.name} has no column(s) {unknown}. It has "
                f"{len(self.columns)}: {list(self.columns)[:12]}"
                f"{' ...' if len(self.columns) > 12 else ''}"
            )
        return wanted

    def batches(
        self, *, columns: Optional[Sequence[str]] = None, batch_rows: int = 0
    ) -> Iterator[pd.DataFrame]:
        """Yield the dataset a chunk at a time, oldest file first."""
        for batch in self.scanner(columns=columns, batch_rows=batch_rows).to_batches():
            if batch.num_rows:
                yield batch.to_pandas()

    def head(
        self, n: int = 1000, *, columns: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """
        The first `n` rows, for looking rather than for computing.

        Bounded on purpose. This is what a tool result can carry and what a
        caller uses to see the shape; anything that has to be right about
        the whole dataset iterates `batches()`.
        """
        if n <= 0:
            raise ValidationError(f"head needs a positive row count, got {n}")
        dataset = open_dataset(self.path, fmt=self.fmt)
        table = dataset.scanner(
            columns=self._checked_columns(columns),
            batch_size=min(int(n), DEFAULT_BATCH_ROWS),
        ).head(int(n))
        return table.to_pandas()


def open_dataset(path: Path, *, fmt: Optional[str] = None):
    """The pyarrow dataset for a file or a directory of them."""
    arrow_dataset = _pyarrow_dataset()
    resolved = path if isinstance(path, Path) else resolve_path(str(path))
    fmt = fmt or _infer_format(resolved)
    if fmt not in FORMATS:
        raise ValidationError(
            f"unknown format {fmt!r}; expected one of {list(FORMATS)}"
        )
    try:
        return arrow_dataset.dataset(str(resolved), format=fmt)
    except Exception as exc:  # noqa: BLE001 -- one refusal, not an arrow trace
        raise ValidationError(
            f"{resolved} could not be opened as a {fmt} dataset -- {exc}. A "
            "directory is read as one partitioned dataset, so every file in "
            "it has to share a schema."
        ) from exc


def inspect(
    path: str,
    *,
    kind: str = "",
    fmt: Optional[str] = None,
    known_rows: Optional[int] = None,
    count_rows: bool = True,
) -> ExternalDataset:
    """
    Read the schema and the file statistics, and nothing else.

    For Parquet this touches footers only, so it is fast on a dataset far
    too large to read. For CSV there is no footer and the row count is a
    SCAN -- that difference is real and is reported rather than hidden,
    because a caller choosing a format deserves to know which one answers
    "how many rows" for free.

    `known_rows` is that difference made survivable. Registration counts
    once and records the number; resolving the same reference passes it back
    rather than paying for the scan again. It is a cache and is treated like
    one: a caller that has reason to doubt it (the fingerprint moved, so the
    bytes are not the ones that were counted) passes `count_rows=True` with
    no `known_rows` and gets a fresh number.
    """
    resolved = resolve_path(path)
    fmt = fmt or _infer_format(resolved)
    dataset = open_dataset(resolved, fmt=fmt)
    schema = dataset.schema
    columns = tuple(str(name) for name in schema.names)
    if not columns:
        raise ValidationError(
            f"{resolved} has no columns. An empty schema is either a "
            "zero-byte file or a directory whose files disagree."
        )
    if known_rows is not None:
        rows: Optional[int] = int(known_rows)
    elif not count_rows:
        rows = None
    else:
        try:
            rows = int(dataset.count_rows())
        except Exception:  # noqa: BLE001 - a count is a convenience, not the point
            rows = None

    files = _files(resolved)
    handle = ExternalDataset(
        path=resolved,
        kind=str(kind),
        fmt=fmt,
        columns=columns,
        dtypes={str(name): str(schema.field(name).type) for name in schema.names},
        rows=rows,
        n_files=len(files),
        size_bytes=total_bytes(resolved),
        fingerprint=fingerprint(resolved),
    )
    return handle


def book_levels(columns: Sequence[str]) -> int:
    """
    How many COMPLETE depth levels a book's columns describe.

    Complete is the operative word. A level with three of its four columns
    is not a level -- `book_metrics` reading a `bid_price_3` with no
    `bid_size_3` would weight the touch against a missing size and report a
    microprice that leans on nothing. Counting stops at the first gap rather
    than counting every level that has any column, so a book with levels
    0-2 and a stray `ask_price_7` reads as three levels deep, which is what
    it is.
    """
    present = set(str(c) for c in columns)
    level = 0
    while all(
        f"{side}_{field}_{level}" in present
        for side in ("bid", "ask")
        for field in ("price", "size")
    ):
        level += 1
    return level


def required_columns(kind: str) -> Tuple[str, ...]:
    if kind not in KIND_COLUMNS:
        raise ValidationError(
            f"unknown external dataset kind {kind!r}; expected one of "
            f"{sorted(KIND_COLUMNS)}. The kind is what makes a mismatched "
            "handoff fail by name rather than several frames deep."
        )
    return KIND_COLUMNS[kind]


#: kind -> the databento normalizer that produces it, for the refusal hint.
_DATABENTO_NORMALIZER: Dict[str, str] = {
    "order_book_panel": "normalize_book",
    "order_event_panel": "normalize_mbo",
    "quote_panel": "normalize_quotes",
    "tick_tape": "normalize_trades",
}


def _looks_like_databento(columns: Sequence[str]) -> bool:
    """Imported lazily so `external` does not depend on the vendor module."""
    try:
        from standard_quant_tools.data.databento import looks_like_databento
    except ImportError:  # pragma: no cover - vendor module always ships
        return False
    return bool(looks_like_databento(columns))


def check_schema(kind: str, columns: Sequence[str]) -> List[str]:
    """
    Column-level problems with this dataset for this kind, as text.

    A refusal here NAMES THE FIX when it can recognize the vendor. A raw
    Databento export fails this check for a boring reason -- it spells the
    same quantity `bid_px_00` where this library spells it `bid_price_0` --
    and "missing column bid_price_0" sends someone hunting for data that is
    right there under another name. Saying which normalizer to run is the
    difference between a dead end and a next step.
    """
    required = required_columns(kind)
    present = set(str(c) for c in columns)
    missing = [c for c in required if c not in present]
    problems: List[str] = []
    if missing:
        hint = ""
        if _looks_like_databento(columns):
            hint = (
                " These columns look like a RAW DATABENTO export, which "
                "spells the same fields differently (`bid_px_00` for "
                "`bid_price_0`, `ts_recv` for `timestamp`) and carries "
                "fixed-point prices and int64-max sentinels. Convert it "
                "first with standard_quant_tools.data.databento."
                f"{_DATABENTO_NORMALIZER.get(kind, 'normalize_book')}(), "
                "which also masks the sentinels and scales the prices -- "
                "registering it unconverted would put $9.2 billion quotes "
                "in your book."
            )
        problems.append(
            f"missing column(s) {missing} that a {kind!r} needs. "
            f"{KIND_DESCRIPTIONS[kind]}{hint}"
        )
    if kind == "order_book_panel" and not missing:
        levels = book_levels(columns)
        if levels < 1:
            problems.append(
                "no complete depth level: a level needs all four of "
                "bid_price_i, bid_size_i, ask_price_i, ask_size_i, and "
                "level 0 does not have them."
            )
    return problems


__all__ = [
    "DEFAULT_BATCH_ROWS",
    "DEFAULT_SCAN_LIMIT",
    "FORMATS",
    "KIND_COLUMNS",
    "KIND_DESCRIPTIONS",
    "ExternalDataset",
    "book_levels",
    "check_schema",
    "fingerprint",
    "inspect",
    "open_dataset",
    "required_columns",
    "resolve_path",
    "total_bytes",
]
