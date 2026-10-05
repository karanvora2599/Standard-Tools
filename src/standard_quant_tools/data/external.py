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

WHERE IT MAY READ. The path is caller-supplied and reaches this library from
an agent, so "read the file at this path and put it in a tool result" is a
capability worth bounding, and it is bounded twice. First by DIRECTORY: a
path is read only when it lies inside the runs directory or a directory the
operator listed in `SQT_EXTERNAL_DIRS` (`external_roots`), checked on the
text before anything touches the filesystem and again after links are
followed, and for a directory dataset on every file inside it. Second by
FORMAT: only Parquet and CSV are read, so even inside the fence a reader that
fails on anything else refuses to be a general file-exfiltration primitive,
and both formats cover what market-data vendors actually ship. The format
bound alone was the whole bound until the CHANGELOG entry of 2026-09-27, and
it still let any market-data-shaped file on the machine -- a blotter, a
positions export -- be previewed row by row.

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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from standard_quant_tools._containment import is_within, is_within_any
from standard_quant_tools._env import env_paths
from standard_quant_tools.error import ValidationError

#: The setting that lists the directories, beyond the runs directory, that
#: external data may be read from and a converted extract written to.
EXTERNAL_DIRS_ENV = "SQT_EXTERNAL_DIRS"

#: Where, under the runs directory, a conversion with a relative `out_path`
#: lands -- and the only part of the runs directory a conversion may write.
EXTRACTS_DIR = "extracts"

#: Two leading separators, in any mix, begin a network share or a device
#: path on Windows (`\\host\share`, `\\.\PhysicalDrive0`). Refused on the
#: text, because even asking whether one exists opens a connection to the
#: host it names and authenticates this machine to it.
_NETWORK_PREFIXES = ("\\\\", "//", "\\/", "/\\")

#: Windows' extended-length prefix in front of a drive letter is a local
#: path spelled long, and is accepted as the path it spells.
_EXTENDED_DRIVE_PREFIX = "\\\\?\\"

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
    # `timestamp` for the same reason `price` was added to the panel above:
    # without it a file satisfies REGISTRATION and then fails inside the
    # estimator. `handoff.KINDS` calls both of these "timestamp-indexed", and
    # a Parquet file carries an index as a column -- so the stamp has to be
    # there under a name, and `timestamp` is the name the normalizers write
    # and `_column_span` reads.
    "tick_tape": ("timestamp", "price", "size"),
    "quote_panel": ("timestamp", "bid_price", "ask_price"),
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
    "tick_tape": (
        "Individual trades: `timestamp`, `price`, `size`. The stamp is a "
        "COLUMN here and an index on the published form of the same kind; "
        "resolving one restores the index the estimators need."
    ),
    "quote_panel": (
        "Top-of-book quotes: `timestamp`, `bid_price`, `ask_price`. The "
        "stamp is a column here, as it is for a tick tape."
    ),
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


# ── the fence ─────────────────────────────────────────────────────────
#
# WHY THE RUNS DIRECTORY AND NOT THE WORKING DIRECTORY. The working
# directory of an MCP server is chosen by the client that launches it --
# often the home directory or `/` -- and moves with `os.chdir`, so a fence
# that included it would cover most of the disk on the machines where it
# matters. The runs directory is where this library's own depth fetches
# register their Parquet, so the built-in chains keep working with nothing
# configured. Anything else is the operator's decision, made in the
# environment the process starts with and never in a tool argument, so an
# agent cannot widen it.
#
# WHY THE CACHE IS NEVER A WRITE TARGET. A Parquet named like a cache entry
# for a window not yet fetched would be served as a hit. The audit
# directory and the rest of the runs directory are refused for the same
# reason: a file placed there is read back as something this library wrote.


def _runs_root() -> Path:
    from standard_quant_tools._runspath import runs_dir

    return runs_dir()


def _resolved(path: Path) -> Path:
    """`path` with links followed, or its absolute form if that fails."""
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, ValueError):
        return Path(os.path.abspath(path))


def _spellings(roots: Sequence[Path]) -> Tuple[Path, ...]:
    """Each root as written and as resolved, for the check on the text.

    A root that is itself a link (`/tmp` on macOS is `/private/tmp`) is
    matched either way before anything is opened; the check after links
    are followed uses the resolved form only.
    """
    spelled: List[Path] = []
    for root in roots:
        for form in (Path(os.path.abspath(root)), _resolved(root)):
            if form not in spelled:
                spelled.append(form)
    return tuple(spelled)


def _unique_resolved(roots: Sequence[Path]) -> Tuple[Path, ...]:
    resolved: List[Path] = []
    for root in roots:
        form = _resolved(root)
        if form not in resolved:
            resolved.append(form)
    return tuple(resolved)


def configured_external_dirs() -> Tuple[Path, ...]:
    """The directories `SQT_EXTERNAL_DIRS` lists, as written (absolute).

    Refuses, by name, a relative entry or one naming a file.
    """
    return env_paths(EXTERNAL_DIRS_ENV)


def external_roots() -> Tuple[Path, ...]:
    """
    Every directory external data may be read from, resolved: the runs
    directory, then each directory `SQT_EXTERNAL_DIRS` lists.

    The runs directory is always one of them, because this library's own
    depth fetches register the Parquet they write there.
    """
    return _unique_resolved((_runs_root(),) + configured_external_dirs())


def _extracts_root() -> Path:
    return _runs_root() / EXTRACTS_DIR


def _roots_text(roots: Sequence[Path]) -> str:
    return os.pathsep.join(str(root) for root in roots) or "none"


def _widen_hint() -> str:
    return (
        f"To use another directory, add it to {EXTERNAL_DIRS_ENV} -- absolute "
        f"paths separated by {os.pathsep!r} -- in the environment this process "
        "starts with. It is read from there and nowhere else, so no tool call "
        "can widen it."
    )


def _read_refusal(text: str, roots: Sequence[Path]) -> ValidationError:
    # The caller's own text is echoed and nothing derived from it: not the
    # resolved form (which would disclose where a link points) and not an
    # expansion. The same sentence whether or not the path exists, so the
    # fence answers no question about what lies outside it.
    return ValidationError(
        f"{text!r} is outside the directories external data may be read from "
        f"({_roots_text(roots)}). This process reads external data only from "
        f"its runs directory and the directories {EXTERNAL_DIRS_ENV} lists. "
        + _widen_hint()
    )


def _write_refusal(text: str) -> ValidationError:
    return ValidationError(
        f"{text!r} is outside the directories a converted extract may be "
        f"written to: the runs directory's {EXTRACTS_DIR!r} folder "
        f"({_extracts_root()}), where a relative name lands, and the "
        f"directories {EXTERNAL_DIRS_ENV} lists "
        f"({_roots_text(configured_external_dirs())}). " + _widen_hint()
    )


def _network_refusal(text: str) -> ValidationError:
    return ValidationError(
        f"{text!r} is a network or device path. Those are refused before "
        "anything opens them, because opening one makes this machine "
        "authenticate to whatever host it names. Copy the data to a local "
        f"directory that {EXTERNAL_DIRS_ENV} lists."
    )


def _local_text(text: str) -> str:
    """The path as the caller spelled it, minus a Windows extended-length
    prefix in front of a drive letter; refuses network and device paths."""
    if "\x00" in text:
        raise ValidationError(
            f"{text!r} contains a NUL character, which no path can hold."
        )
    local = text
    rest = text[len(_EXTENDED_DRIVE_PREFIX) :]
    if (
        text.startswith(_EXTENDED_DRIVE_PREFIX)
        and len(rest) >= 2
        and rest[0].isalpha()
        and rest[1] == ":"
    ):
        local = rest
    if local[:2] in _NETWORK_PREFIXES:
        raise _network_refusal(text)
    expanded = os.path.expanduser(local)
    if expanded[:2] in _NETWORK_PREFIXES:
        raise _network_refusal(text)
    return expanded


def _follow(lexical: Path, text: str) -> Path:
    try:
        return lexical.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValidationError(
            f"{text!r} could not be resolved ({type(exc).__name__}). Pass a "
            "plain absolute path to a file or directory."
        ) from exc


def _fenced_read(text: str) -> Path:
    """The fence for a read, before anything is known about the target.

    The order is the point. The TEXT is checked first, so a network share,
    another drive or a `..` walk out of a root is refused without the
    filesystem being asked anything. Links are followed second and the
    answer checked again, so a link or junction inside a root that points
    out is refused. Only a path inside the fence is ever asked whether it
    exists.
    """
    roots = external_roots()
    lexical = Path(os.path.abspath(_local_text(text)))
    if not is_within_any(lexical, _spellings(roots)):
        raise _read_refusal(text, roots)
    resolved = _follow(lexical, text)
    if not is_within_any(resolved, roots):
        raise _read_refusal(text, roots)
    return resolved


def _vetted_files(directory: Path, roots: Sequence[Path]) -> List[Path]:
    """Every file a directory dataset holds, each confirmed inside the fence.

    The directory being inside is not enough: the walk follows file links
    and junctions, and so does the reader, so one link planted among the
    partitions would put a file from anywhere into the dataset.

    Every entry the walk lists is checked, directories included, and the
    files are taken from that one walk. On Windows the walk descends into a
    junction and meets the files behind it; on POSIX it lists a directory
    symlink without descending, so checking files alone let such a link
    pass in silence there while Windows refused it.
    """
    entries = sorted(directory.rglob("*")) if directory.is_dir() else [directory]
    for entry in entries:
        target = _resolved(entry)
        if not is_within_any(target, roots):
            relative = entry.relative_to(directory) if entry != directory else entry
            raise ValidationError(
                f"{directory} holds {str(relative)!r}, which is a link to a "
                "location outside the directories external data may be read from "
                f"({_roots_text(roots)}). A directory is read file by file, so "
                "every file in it has to lie inside them too. Remove the link, "
                "or copy the file into the directory. " + _widen_hint()
            )
    return [entry for entry in entries if entry.is_file()]


def _resolve_readable(path: str) -> Tuple[Path, List[Path]]:
    """`resolve_path`, plus the vetted file list the reader is given."""
    text = str(path).strip()
    if not text:
        raise ValidationError(
            "an external dataset needs a path; got an empty string. Pass the "
            "file, or the directory holding a partitioned dataset."
        )
    resolved = _fenced_read(text)
    if not resolved.exists():
        raise ValidationError(
            f"no file or directory at {resolved}. Nothing is copied when a "
            "dataset is registered, so the path has to be readable from "
            "wherever this library runs, not only from where it was typed."
        )
    if resolved.is_dir():
        return resolved, _vetted_files(resolved, external_roots())
    return resolved, [resolved]


def resolve_path(path: str) -> Path:
    """
    Turn a caller-supplied path into one that exists inside the fence, or
    say why not.

    Every read of external data comes through here -- registration, the
    re-read on every `resolve()` and `describe()` of a registered reference,
    a vendor conversion's input and an external model panel -- so this is
    the one place the fence has to hold. Narrowing `SQT_EXTERNAL_DIRS`
    therefore revokes access to what was registered under the wider value:
    a reference whose file now lies outside stops resolving.

    `~` is expanded, because the fence decides where it may point.
    Environment variables are NOT: a model-chosen `$NAME` or `%NAME%` would
    otherwise put that variable's value into a refusal, and a refusal is
    written verbatim into the decision log.
    """
    return _resolve_readable(path)[0]


def resolve_output_path(path: str) -> Path:
    """
    Where a conversion may write `path`, resolved, or a refusal.

    A relative name lands under the runs directory's `extracts` folder,
    never in the working directory. An absolute path has to lie inside that
    folder or a directory `SQT_EXTERNAL_DIRS` lists, and never inside the
    OHLCV cache, the audit directory or the rest of the runs directory --
    even when a listed directory contains them. Whether something is
    already at the path is the caller's check, made after this one.
    """
    text = str(path).strip()
    if not text:
        raise ValidationError(
            "a conversion needs an out_path; got an empty string. Give a file "
            f"name, which lands in the runs directory's {EXTRACTS_DIR!r} folder."
        )
    local = Path(_local_text(text))
    extracts = _extracts_root()
    if not local.is_absolute():
        local = extracts / local
    configured = configured_external_dirs()
    allowed = (extracts,) + configured
    lexical = Path(os.path.abspath(local))
    if not is_within_any(lexical, _spellings(allowed)):
        raise _write_refusal(text)
    resolved = _follow(lexical, text)
    if not is_within_any(resolved, _unique_resolved(allowed)):
        raise _write_refusal(text)
    _refuse_owned_stores(resolved, text, extracts)
    return resolved


def _owned_stores() -> List[Tuple[Path, str, str]]:
    """The directories this library reads back as its own, with what a
    planted file would be mistaken for."""
    from standard_quant_tools.audit.paths import _audit_dir
    from standard_quant_tools.data import _cache

    stores = []
    # The cache root the cache itself is using: frozen at import, and what a
    # test that relocates the cache assigns.
    cache_root = getattr(_cache, "_CACHE_ROOT", None)
    if cache_root is None and callable(getattr(_cache, "cache_root", None)):
        cache_root = _cache.cache_root()
    if cache_root is not None:
        stores.append(
            (
                Path(cache_root),
                "the OHLCV cache directory",
                "a cache hit for a window nobody fetched",
            )
        )
    stores.append((_audit_dir(), "the audit directory", "part of the decision record"))
    stores.append(
        (
            _runs_root(),
            f"the runs directory outside its {EXTRACTS_DIR!r} folder",
            "an artifact nothing published",
        )
    )
    return stores


def _inside(resolved: Path, store: Path) -> bool:
    """Whether `resolved` lies inside `store`, asking the filesystem when
    the spelling alone cannot say.

    Path comparison is case-sensitive on macOS while its filesystem usually
    is not, so `Cache/x` and `cache/x` can be one directory spelled two
    ways. For a store this library owns, the wrong answer lets a write in,
    so an existing ancestor is compared by identity as well as by name.
    """
    if is_within(resolved, store):
        return True
    try:
        store_stat = os.stat(store)
    except OSError:
        return False
    if not store_stat.st_ino:  # a filesystem with no file identity to compare
        return False
    for ancestor in (resolved, *resolved.parents):
        try:
            if os.path.samestat(os.stat(ancestor), store_stat):
                return True
        except OSError:
            continue
    return False


def _refuse_owned_stores(resolved: Path, text: str, extracts: Path) -> None:
    extracts_resolved = _resolved(extracts)
    in_extracts = is_within(resolved, extracts_resolved)
    for root, label, mistaken_for in _owned_stores():
        store = _resolved(root)
        if not _inside(resolved, store):
            continue
        # The extracts folder is the one place a conversion is meant to
        # write, so a store that CONTAINS it (the runs directory always; a
        # cache configured as a parent of the runs directory) does not
        # claim what lands inside it.
        if in_extracts and is_within(extracts_resolved, store):
            continue
        raise ValidationError(
            f"{text!r} lies inside {label}, which this library owns. A "
            "converted extract is never written there, even when "
            f"{EXTERNAL_DIRS_ENV} covers it, because a file placed there would "
            f"be read back as {mistaken_for}. Give a bare file name, which "
            f"lands in {extracts}, or a path in a directory "
            f"{EXTERNAL_DIRS_ENV} lists."
        )


def _files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file())


def _reader_files(directory: Path, files: Sequence[Path]) -> List[str]:
    """The files of a directory the reader is handed, by name.

    The same ones the reader's own directory discovery would pick -- it
    skips any name starting with `.` or `_` (`_SUCCESS`, `.crc`) -- but
    taken from the list the fence already checked, so a link planted after
    the check is not picked up by a second walk.
    """
    chosen = []
    for file in files:
        parts = file.relative_to(directory).parts
        if any(part.startswith((".", "_")) for part in parts):
            continue
        chosen.append(str(file))
    return chosen


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
    """
    The pyarrow dataset for a file or a directory of them.

    Fenced on every call, a `Path` included: a handle outlives the check
    made when it was created, and this is where bytes are actually read.
    """
    resolved, files = _resolve_readable(str(path))
    return _open(resolved, files, fmt)


def _open(resolved: Path, files: Sequence[Path], fmt: Optional[str]):
    arrow_dataset = _pyarrow_dataset()
    fmt = fmt or _infer_format(resolved)
    if fmt not in FORMATS:
        raise ValidationError(
            f"unknown format {fmt!r}; expected one of {list(FORMATS)}"
        )
    source: Any = _reader_files(resolved, files) if resolved.is_dir() else str(resolved)
    try:
        return arrow_dataset.dataset(source, format=fmt)
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
    resolved, vetted = _resolve_readable(path)
    fmt = fmt or _infer_format(resolved)
    dataset = _open(resolved, vetted, fmt)
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

    handle = ExternalDataset(
        path=resolved,
        kind=str(kind),
        fmt=fmt,
        columns=columns,
        dtypes={str(name): str(schema.field(name).type) for name in schema.names},
        rows=rows,
        n_files=len(vetted),
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


#: Spellings that satisfy a required column.
#:
#: `__index_level_0__` is what pandas names an unnamed index when it writes
#: Parquet, so a tape written from a timestamp-indexed frame carries its stamp
#: under that name and no `timestamp` column at all. Refusing it would be
#: refusing a readable file over the spelling of something that is in it --
#: and `handoff._time_indexed` reads exactly these two names, in this order.
#:
#: This is a NAME check. An unnamed index can hold a string or a counter just
#: as easily as an instant, and `check_schema` is handed column names without
#: their types; one that is not an instant is refused on resolve instead, by a
#: message that lists the columns it did find.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "timestamp": ("__index_level_0__",),
}


def _satisfied(column: str, present: set) -> bool:
    """Is a required column present under its own name or an accepted one?"""
    if column in present:
        return True
    return any(alias in present for alias in COLUMN_ALIASES.get(column, ()))


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
    missing = [c for c in required if not _satisfied(c, present)]
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
    "EXTERNAL_DIRS_ENV",
    "EXTRACTS_DIR",
    "FORMATS",
    "KIND_COLUMNS",
    "KIND_DESCRIPTIONS",
    "ExternalDataset",
    "book_levels",
    "check_schema",
    "configured_external_dirs",
    "external_roots",
    "fingerprint",
    "inspect",
    "open_dataset",
    "required_columns",
    "resolve_output_path",
    "resolve_path",
    "total_bytes",
]
