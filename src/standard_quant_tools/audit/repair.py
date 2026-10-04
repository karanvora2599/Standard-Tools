"""Cutting a torn final line off the newest day file or the chain index --
what `sqt audit repair-tail` runs.

A write cut short by a crash or a full disk leaves the file it was writing
ending in a fragment: bytes with no newline after them that are not a whole
line. The writer appends a record to the newest day, and appends an entry to
the chain index just before the first record of a new day, so either can be
left that way. The writer refuses to append after a fragment, because
nothing can chain onto one, and names this as the remedy. Until now the
remedy was a manual truncation at a byte offset, which is easy to get one
byte wrong and leaves no trace of what was removed -- and for the chain
index it still was.

`repair_torn_tails` makes that cut and nothing else. By default it only
reports what it would cut. With `confirm=True` it takes the locks the writer
takes, re-reads each file under them, saves each fragment's bytes to a side
file beside it and truncates it to where the fragment starts -- so each file
is byte for byte what it was before the interrupted write, and the trail
verifies as it did then. `repair_torn_tail` does the same for the newest day
alone, and `repair_torn_index` for the chain index alone.

It refuses everything that is not that one situation, because each of the
others calls for a person rather than a cut:

- the final line is complete (with or without its newline -- the writer
  restores a lost newline itself), so there is nothing to cut;
- the final line is damaged but ends in a newline, or parses to something
  that is not a record or an entry, or begins with a complete one: a write
  cut short leaves none of these;
- another line of the file is unreadable too, so the damage is not only at
  the end and cutting the tail would not make the file whole -- a corrupt
  line in the middle is evidence, not something to repair;
- the day is not the newest: only the day being written can be cut short,
  so a fragment at the end of an earlier day was left some other way.

Both files are examined before either is touched, and either one's refusal
leaves both as they were.

CLI ONLY. Like `gc`, `seal_day`, `hold_day` and signing, this changes the
trail, and an agent able to cut its own decision log is not audited by it;
it is in no dispatch table, and a test pins that.

Local filesystem only, like `verify` and `retention`: it reads and truncates
the files directly rather than through a storage backend.
"""

import json
import os
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

from standard_quant_tools import _filelock
from standard_quant_tools.error import ValidationError

from .paths import _DAY_FILE_RE, _INDEX_FILENAME, _audit_dir, _iter_day_files
from .storage import lock_path_for
from .verify import _json_kind, _read_line

#: What a side file's name adds to the file's: `2026-09-28.jsonl.torn-...`,
#: `_chain_index.jsonl.torn-...`. Beside the file and named for it, like the
#: `.hold` sidecar, and never a `*.jsonl` name, so no reader of day files or
#: of the index takes it for one.
TORN_SUFFIX = ".torn-"


@dataclass(frozen=True)
class TornTail:
    """A torn final line: where it starts, its bytes, and -- once cut --
    the side file they were saved to."""

    #: The file the line ends: a day file, or the chain index.
    day: Path
    #: The byte the fragment starts at, which is the file's size after the cut.
    offset: int
    fragment: bytes
    #: Where the fragment was saved; None while it has only been reported.
    side_file: Optional[Path] = None

    @property
    def cut(self) -> bool:
        return self.side_file is not None

    @property
    def is_index(self) -> bool:
        """True for the chain index, False for a day file."""
        return self.day.name == _INDEX_FILENAME

    @property
    def what(self) -> str:
        """What a whole line of the file is: "entry" or "record"."""
        return "entry" if self.is_index else "record"


class _NothingTorn(ValidationError):
    """A file whose final line is whole, or that holds nothing: nothing to
    cut, and nothing wrong with its end. Kept apart from the refusals that
    find damage, so the command can examine the day and the index and act
    on whichever one is torn."""

    def __init__(self, label: Optional[str], why: str, remedy: str = "") -> None:
        #: The file's name, or None when there is no file to name.
        self.label = label
        self.why = why
        self.remedy = remedy
        said = f"{label}: {why} Nothing was cut." if label else why
        super().__init__(f"{said} {remedy}".rstrip())

    @property
    def said(self) -> str:
        return f"{self.label}: {self.why}" if self.label else self.why


def _refuse(day: Path, why: str, remedy: str = "") -> ValidationError:
    return ValidationError(f"{day.name}: {why} Nothing was cut. {remedy}".rstrip())


_BY_HAND = (
    "`sqt verify` names the line; copy the file somewhere safe and examine "
    "it by hand."
)


def _last_nonblank(lines: List[bytes]) -> Tuple[int, bytes]:
    for number in range(len(lines), 0, -1):
        if lines[number - 1].strip():
            return number, lines[number - 1]
    return 0, b""


def _find_torn_tail(path: Path, what: str = "record") -> Tuple[int, bytes]:
    """`(offset, fragment)` when the file's final line is torn; refuses,
    naming why, when it is not. `what` is what a whole line of the file
    is: "record" for a day, "entry" for the chain index."""
    a_what = f"an {what}" if what[0] in "aeiou" else f"a {what}"
    whats = "entries" if what == "entry" else f"{what}s"
    whole = "index" if what == "entry" else "day"
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise _refuse(path, "the file does not exist.", _BY_HAND) from None
    if not data.strip():
        raise _NothingTorn(
            path.name, f"it holds no {what}, so there is nothing to cut."
        )
    offset = data.rfind(b"\n") + 1
    fragment = data[offset:]

    if not fragment.strip():
        number, last = _last_nonblank(data.split(b"\n"))
        parsed, reason = _read_line(last)
        if parsed is not None:
            raise _NothingTorn(
                path.name,
                f"the final line is a complete {what}, so there is nothing "
                "torn to cut.",
                "If the writer refuses to append, the damage is elsewhere: " + _BY_HAND,
            )
        raise _refuse(
            path,
            f"line {number}, the last, is damaged ({reason}) but ends in a "
            "newline, which a write cut short never leaves after a partial "
            f"{what} -- this is not a torn write.",
            _BY_HAND,
        )

    try:
        text: Optional[str] = fragment.decode("utf-8").strip()
    except UnicodeDecodeError:
        # Cut inside a multi-byte character: a prefix of a line all the
        # same, and nothing that parses.
        text = None
    if text is not None:
        try:
            value, end = json.JSONDecoder().raw_decode(text)
        except ValueError:
            pass  # no complete JSON value at its start: a torn line
        else:
            if end == len(text) and isinstance(value, dict):
                raise _NothingTorn(
                    path.name,
                    f"the final line is a complete {what} that has lost only "
                    "its newline; the writer writes that newline itself "
                    f"before the next {what}, so there is nothing to cut.",
                )
            if end == len(text):
                raise _refuse(
                    path,
                    f"the final line is a JSON {_json_kind(value)}, not "
                    f"{a_what}, and no write of {a_what} is cut short into one.",
                    _BY_HAND,
                )
            raise _refuse(
                path,
                "the final line begins with a complete JSON value and goes "
                "on past it, so cutting the line would remove more than a "
                "fragment.",
                _BY_HAND,
            )

    earlier = data[:offset].split(b"\n")
    unreadable = [
        number
        for number, raw in enumerate(earlier, start=1)
        if raw.strip() and _read_line(raw)[0] is None
    ]
    if unreadable:
        raise _refuse(
            path,
            f"line(s) {unreadable} are not readable {whats} either, so the "
            "damage is not only at the end and cutting the final fragment "
            f"would not make the {whole} whole.",
            _BY_HAND,
        )
    return offset, fragment


def _day_to_repair(date: Optional[str], directory: Path) -> Path:
    """The newest day file, which `date` (when given) has to name."""
    if date is not None and not _DAY_FILE_RE.match(f"{date}.jsonl"):
        raise ValidationError(
            f"{date!r} is not a calendar day; pass it as YYYY-MM-DD, or omit "
            "it to repair the newest day."
        )
    days = _iter_day_files(directory)
    if not days:
        # Nothing to examine; with no date asked for, the chain index may
        # still be, so this is not a refusal of the trail.
        nothing = _NothingTorn(
            None,
            f"there is no day file in {directory}, so there is nothing to repair.",
            "Check SQT_AUDIT_DIR.",
        )
        if date is None:
            raise nothing
        raise ValidationError(str(nothing))
    newest = days[-1]
    if date is None or date == newest.stem:
        return newest
    if not (directory / f"{date}.jsonl").exists():
        raise ValidationError(
            f"there is no day file for {date} in {directory}; the newest day "
            f"is {newest.stem}."
        )
    raise ValidationError(
        f"{date} is not the newest day ({newest.stem}). Nothing was cut: a "
        "write is only ever cut short on the day being written, so a "
        "fragment at the end of an earlier day was left some other way, and "
        "cutting it would hide how. " + _BY_HAND
    )


def _side_file(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return path.with_name(f"{path.name}{TORN_SUFFIX}{stamp}-{uuid.uuid4().hex[:8]}")


def _index_path(directory: Path) -> Path:
    return directory / _INDEX_FILENAME


def _find_torn_index(directory: Path) -> Tuple[Path, int, bytes]:
    index = _index_path(directory)
    if not index.exists():
        raise _NothingTorn(
            index.name,
            f"there is no chain index in {directory}, so there is nothing to cut.",
        )
    offset, fragment = _find_torn_tail(index, "entry")
    return index, offset, fragment


def _find_torn_day(date: Optional[str], directory: Path) -> Tuple[Path, int, bytes]:
    day = _day_to_repair(date, directory)
    offset, fragment = _find_torn_tail(day, "record")
    return day, offset, fragment


def _cut_all(found: List[Tuple[Path, int, bytes]]) -> List[TornTail]:
    """Cut each fragment, every file already opened for writing, so a file
    that cannot be written refuses before any is cut."""
    handles = []
    try:
        for path, _offset, _fragment in found:
            try:
                handles.append(open(path, "r+b"))
            except OSError as exc:
                raise _refuse(
                    path,
                    f"the file cannot be opened for writing ({exc}).",
                    "A sealed day is read-only; make it writable, then retry.",
                ) from exc
        cut: List[TornTail] = []
        for (path, offset, fragment), handle in zip(found, handles):
            side_file = _side_file(path)
            with open(side_file, "xb") as saved:
                saved.write(fragment)
                saved.flush()
                os.fsync(saved.fileno())
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())
            cut.append(TornTail(path, offset, fragment, side_file))
        return cut
    finally:
        for handle in handles:
            handle.close()


def repair_torn_tail(
    date: Optional[str] = None,
    audit_dir: Optional[Union[str, Path]] = None,
    confirm: bool = False,
) -> TornTail:
    """
    Find the torn final line of the newest day (`date`, "YYYY-MM-DD", which
    has to be the newest; omitted, the newest), and with `confirm=True` cut
    it off.

    Without `confirm` nothing is written: the result says where the
    fragment starts and what it holds. With it, under the lock the writer
    holds for that day, the day is read again and checked again, the
    fragment's bytes are written to a new side file beside the day
    (`<date>.jsonl.torn-<UTC time>-<id>`) and flushed to disk, and only
    then is the day truncated to where the fragment starts. The result
    names the side file.

    Refuses with a ValidationError naming the reason and the remedy: no
    day file, a day that is not the newest, a final line that is complete
    or damaged in a way a cut-short write does not leave, other unreadable
    lines in the day, a lock that cannot be taken, and a day that cannot
    be written (a sealed day is read-only).
    """
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    day, offset, fragment = _find_torn_day(date, directory)
    if not confirm:
        return TornTail(day=day, offset=offset, fragment=fragment)

    with _filelock.exclusive(
        lock_path_for(day),
        required=True,
        purpose=f"cutting the torn final line of {day.name}",
    ):
        # Everything again, under the lock: what was reported before is
        # not what is cut unless it is still there.
        (torn,) = _cut_all([_find_torn_day(day.stem, directory)])
    return torn


def repair_torn_index(
    audit_dir: Optional[Union[str, Path]] = None, confirm: bool = False
) -> TornTail:
    """
    Find the torn final line of the chain index, and with `confirm=True`
    cut it off: the same report, lock, side file and refusals as
    `repair_torn_tail`, for `_chain_index.jsonl`.

    The writer appends a new day's index entry just before that day's first
    record, so a write cut short there leaves the index torn and the day
    without a file; the cut leaves the index as it was before that write.
    """
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    index, offset, fragment = _find_torn_index(directory)
    if not confirm:
        return TornTail(day=index, offset=offset, fragment=fragment)
    with _filelock.exclusive(
        lock_path_for(index),
        required=True,
        purpose=f"cutting the torn final line of {index.name}",
    ):
        (torn,) = _cut_all([_find_torn_index(directory)])
    return torn


def _examined(find) -> Tuple[Optional[Tuple[Path, int, bytes]], Optional[_NothingTorn]]:
    """`(found, None)` for a torn file, `(None, why)` for one with nothing
    torn; any other refusal is raised."""
    try:
        return find(), None
    except _NothingTorn as whole:
        return None, whole


def repair_torn_tails(
    date: Optional[str] = None,
    audit_dir: Optional[Union[str, Path]] = None,
    confirm: bool = False,
) -> List[TornTail]:
    """
    The newest day (`date`, as `repair_torn_tail` takes it) and the chain
    index, each examined for a torn final line; whichever has one is
    reported, or with `confirm=True` cut, as `repair_torn_tail` and
    `repair_torn_index` cut it. The day comes first in the result.

    Both are examined before anything is written, and a refusal of either
    -- damage a cut-short write does not leave, a day that is not the
    newest -- leaves both as they were. When neither ends in a torn line,
    the refusal says what each ends in. With `confirm`, the index's lock is
    taken before the day's (the writer takes the index's before an earlier
    day's), both files are examined again under them, and both are opened
    for writing before either is cut.
    """
    directory = Path(audit_dir) if audit_dir else _audit_dir()

    def find_all() -> List[Tuple[Path, int, bytes]]:
        day, day_whole = _examined(lambda: _find_torn_day(date, directory))
        index, index_whole = _examined(lambda: _find_torn_index(directory))
        found = [f for f in (day, index) if f is not None]
        if day_whole is not None and index_whole is not None:
            raise ValidationError(
                f"{day_whole.said} {index_whole.said} Nothing was cut. "
                f"{day_whole.remedy}".rstrip()
            )
        return found

    found = find_all()
    if not confirm:
        return [TornTail(path, offset, fragment) for path, offset, fragment in found]

    # The index first, then the day: the order the writer takes them in when
    # it reads the previous day's tail while holding the index.
    order = sorted(found, key=lambda f: f[0].name != _INDEX_FILENAME)
    with ExitStack() as locks:
        for path, _offset, _fragment in order:
            locks.enter_context(
                _filelock.exclusive(
                    lock_path_for(path),
                    required=True,
                    purpose=f"cutting the torn final line of {path.name}",
                )
            )
        found_again = find_all()
        if [f[0] for f in found_again] != [f[0] for f in found]:
            raise ValidationError(
                "the trail changed while the locks were taken: "
                f"{[f[0].name for f in found]} ended in a torn line, and now "
                f"{[f[0].name for f in found_again]} do. Nothing was cut; run "
                "the command again to see what it would cut now."
            )
        return _cut_all(found_again)


__all__ = [
    "TORN_SUFFIX",
    "TornTail",
    "repair_torn_index",
    "repair_torn_tail",
    "repair_torn_tails",
]
