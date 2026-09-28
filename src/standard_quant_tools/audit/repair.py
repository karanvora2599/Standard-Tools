"""Cutting a torn final record off the newest day file -- what
`sqt audit repair-tail` runs.

A write cut short by a crash or a full disk leaves the newest day ending in
a fragment: bytes with no newline after them that are not a record. The
writer refuses to append after it, because nothing can chain onto a
fragment, and names this as the remedy. Until now the remedy was a manual
truncation at a byte offset, which is easy to get one byte wrong and leaves
no trace of what was removed.

`repair_torn_tail` makes that cut and nothing else. By default it only
reports what it would cut. With `confirm=True` it takes the lock the writer
takes, re-reads the day under it, saves the fragment's bytes to a side file
beside the day and truncates the day to where the fragment starts -- so the
day is byte for byte what it was before the interrupted write, and verifies
as it did then.

It refuses everything that is not that one situation, because each of the
others calls for a person rather than a cut:

- the final line is a complete record (with or without its newline -- the
  writer restores a lost newline itself), so there is nothing to cut;
- the final line is damaged but ends in a newline, or parses to something
  that is not a record, or begins with a complete record: a write cut short
  leaves none of these;
- another line of the day is unreadable too, so the damage is not only at
  the end and cutting the tail would not make the day whole;
- the day is not the newest: only the day being written can be cut short,
  so a fragment at the end of an earlier day was left some other way.

CLI ONLY. Like `gc`, `seal_day`, `hold_day` and signing, this changes the
trail, and an agent able to cut its own decision log is not audited by it;
it is in no dispatch table, and a test pins that.

Local filesystem only, like `verify` and `retention`: it reads and truncates
the day file directly rather than through a storage backend.
"""

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

from standard_quant_tools import _filelock
from standard_quant_tools.error import ValidationError

from .paths import _DAY_FILE_RE, _audit_dir, _iter_day_files
from .storage import lock_path_for
from .verify import _json_kind, _read_line

#: What a side file's name adds to the day's: `2026-09-28.jsonl.torn-...`.
#: Beside the day and named for it, like the `.hold` sidecar, and never a
#: `*.jsonl` name, so no reader of day files takes it for one.
TORN_SUFFIX = ".torn-"


@dataclass(frozen=True)
class TornTail:
    """A torn final line: where it starts, its bytes, and -- once cut --
    the side file they were saved to."""

    day: Path
    #: The byte the fragment starts at, which is the day's size after the cut.
    offset: int
    fragment: bytes
    #: Where the fragment was saved; None while it has only been reported.
    side_file: Optional[Path] = None

    @property
    def cut(self) -> bool:
        return self.side_file is not None


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


def _find_torn_tail(day: Path) -> Tuple[int, bytes]:
    """`(offset, fragment)` when the day's final line is torn; refuses,
    naming why, when it is not."""
    try:
        data = day.read_bytes()
    except FileNotFoundError:
        raise _refuse(day, "the file does not exist.", _BY_HAND) from None
    if not data.strip():
        raise _refuse(day, "it holds no record, so there is nothing to cut.")
    offset = data.rfind(b"\n") + 1
    fragment = data[offset:]

    if not fragment.strip():
        number, last = _last_nonblank(data.split(b"\n"))
        record, reason = _read_line(last)
        if record is not None:
            raise _refuse(
                day,
                "the final line is a complete record, so there is nothing "
                "torn to cut.",
                "If the writer refuses to append, the damage is elsewhere: " + _BY_HAND,
            )
        raise _refuse(
            day,
            f"line {number}, the last, is damaged ({reason}) but ends in a "
            "newline, which a write cut short never leaves after a partial "
            "record -- this is not a torn write.",
            _BY_HAND,
        )

    try:
        text: Optional[str] = fragment.decode("utf-8").strip()
    except UnicodeDecodeError:
        # Cut inside a multi-byte character: a prefix of a record all the
        # same, and nothing that parses.
        text = None
    if text is not None:
        try:
            value, end = json.JSONDecoder().raw_decode(text)
        except ValueError:
            pass  # no complete JSON value at its start: a torn record
        else:
            if end == len(text) and isinstance(value, dict):
                raise _refuse(
                    day,
                    "the final line is a complete record that has lost only "
                    "its newline; the writer writes that newline itself "
                    "before the next record, so there is nothing to cut.",
                )
            if end == len(text):
                raise _refuse(
                    day,
                    f"the final line is a JSON {_json_kind(value)}, not a "
                    "record, and no write of a record is cut short into one.",
                    _BY_HAND,
                )
            raise _refuse(
                day,
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
            day,
            f"line(s) {unreadable} are not readable records either, so the "
            "damage is not only at the end and cutting the final fragment "
            "would not make the day whole.",
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
        raise ValidationError(
            f"there is no day file in {directory}, so there is nothing to "
            "repair. Check SQT_AUDIT_DIR."
        )
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


def _side_file(day: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return day.with_name(f"{day.name}{TORN_SUFFIX}{stamp}-{uuid.uuid4().hex[:8]}")


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
    day = _day_to_repair(date, directory)
    if not confirm:
        offset, fragment = _find_torn_tail(day)
        return TornTail(day=day, offset=offset, fragment=fragment)

    with _filelock.exclusive(
        lock_path_for(day),
        required=True,
        purpose=f"cutting the torn final line of {day.name}",
    ):
        # Everything again, under the lock: what was reported before is
        # not what is cut unless it is still there.
        day = _day_to_repair(day.stem, directory)
        offset, fragment = _find_torn_tail(day)
        try:
            handle = open(day, "r+b")
        except OSError as exc:
            raise _refuse(
                day,
                f"the file cannot be opened for writing ({exc}).",
                "A sealed day is read-only; make it writable, then retry.",
            ) from exc
        with handle:
            side_file = _side_file(day)
            with open(side_file, "xb") as saved:
                saved.write(fragment)
                saved.flush()
                os.fsync(saved.fileno())
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())
    return TornTail(day=day, offset=offset, fragment=fragment, side_file=side_file)


__all__ = ["TORN_SUFFIX", "TornTail", "repair_torn_tail"]
