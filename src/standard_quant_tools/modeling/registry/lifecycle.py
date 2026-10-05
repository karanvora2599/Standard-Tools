"""
Lifecycle stages: where a model is on the way from fitted to trusted.

WHY A LOG AND NOT A FIELD. The manifest is content-hashed and immutable --
that is the property every integrity check here rests on -- so a stage
cannot be a field on it without either rewriting the commit point or
signing the stage into the hashes it would then be changing. A stage is a
DECISION made after registration, by someone, for a reason, on evidence,
and possibly reversed. That is a record with a history, not a value: it
lives in `promotions.jsonl` beside the manifest, append-only, and the
current stage is whatever the last line says. Nothing about the model's
identity changes when it is promoted.

WHAT "APPEND-ONLY" IS WORTH. It was a convention of the writer and this
docstring claimed it as a property of the file: "nothing about its
promotion history can be edited without the edit showing". Nothing would
have shown it -- there was no digest and no chain here, and the
`torn_fragments` machinery below detects a TRUNCATED WRITE, not an edit.
The line someone would want gone is the one `promote_model` writes on
purpose, `package_check_waived`, and it sat in the same unprotected file as
the history it was protecting.

Each record now carries `prev`, the digest of the previous line's bytes,
rooted in the manifest digest -- the package's commit point and the one
file a signature covers. `verify_promotion_chain` recomputes it. An
edited, deleted or reordered line is detected, because every later link
would have to be recomputed too.

TWO LIMITS, STATED because the defect this replaces was an overclaim and a
smaller overclaim would be the same mistake. The chain commits to every
record EXCEPT THE MOST RECENT ONE: nothing in the file hashes the tip, so
an edit to the last decision is invisible until another is appended on top
of it. And a log rewritten END TO END, every link recomputed, is not
detected either -- the manifest digest is readable, so the root can be
reproduced. Both need a witness kept outside the package, which this module
does not have and does not claim; `audit/writer.py` keeps a cross-day
chain-index witness log for the same reason. A record written before the
chain existed carries no `prev` and is reported as unchained rather than
as broken -- it is a log from before the property, not a damaged one.

THE STAGES. `candidate` is what registration produces: a model that has
been fitted and walk-forward validated, which is a fact about the fit and
not a judgement about the evidence. `validated` says someone read the
evidence and accepted it. `staging` and `production` are deployment
states. `archived` is terminal. A promotion moves one stage forward at a
time -- skipping `validated` on the way to `production` is the decision
this exists to make visible -- while a demotion may go back to any earlier
live stage, because rolling back is also a decision worth recording.

ONE DECISION AT A TIME. A promotion reads the current stage, checks the
move is allowed and appends it, and nothing used to serialise those three
steps: two callers could both validate against the same old stage and both
append, leaving a log in which a model was archived and live in staging at
once, with two records claiming the same `from_stage`. `promote` now holds
a cross-process lock on a dot-prefixed file beside the log for all three,
re-reading the stage inside it, and refuses rather than proceed when no
lock can be taken -- promotions are rare, and an unserialised one is the
race itself.

A LINE IS COMMITTED BY ITS NEWLINE. A crash part-way through an append
leaves an unterminated fragment at the end of the log, and reading that as
an edit made the model's stage unreadable and every later promotion fail.
An unterminated final line that does not parse is therefore not a record:
it is ignored, copied to a dot-prefixed `.promotions.torn-*` file as
evidence and cut off, under the lock. A line that does not parse anywhere
else is still an edit, and still makes the stage untrustworthy.

A REPAIR IS REPORTED, not only logged. The warning went to a logger the
package gives only a NullHandler, so the one reader who needed it -- whoever
decides whether to trust the model's history -- never saw that a decision
may have been interrupted. `torn_fragments` names every fragment set aside
beside the log, and `inspect_model` and `promote_model` carry it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from standard_quant_tools import _filelock
from standard_quant_tools.error import ValidationError

from .. import artifacts as _artifacts
from . import mirror as _mirror

logger = logging.getLogger(__name__)

STAGES = ("candidate", "validated", "staging", "production", "archived")
LifecycleStage = Literal["candidate", "validated", "staging", "production", "archived"]
_ORDER = {stage: position for position, stage in enumerate(STAGES)}
PROMOTIONS_FILE = "promotions.jsonl"
INITIAL_STAGE = "candidate"
#: The lock beside the log. Dot-prefixed, so a store listing -- and with it
#: package verification, mirroring and pulling -- never sees it.
PROMOTIONS_LOCK_FILE = ".promotions.lock"
#: Where an unterminated fragment cut off the end of the log is kept.
_TORN_PREFIX = ".promotions.torn-"


@dataclass(frozen=True)
class Promotion:
    """One line of the log: a decision, its reason and what it rested on."""

    from_stage: str
    to_stage: str
    reason: str
    actor: str
    timestamp_utc: str
    evidence: List[str]
    #: The digest of the previous line's bytes, or the manifest digest for
    #: the first record. None for a record written before the log was
    #: chained, which `verify_promotion_chain` reports as unchained rather
    #: than broken.
    prev: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _log_path(model_id: str):
    directory = _artifacts.run_dir(model_id)
    if not (directory / "manifest.json").exists():
        raise ValidationError(f"no registered model with model_id={model_id!r}")
    return directory / PROMOTIONS_FILE


def promotions_lock(model_id: str) -> "contextlib.AbstractContextManager[Any]":
    """The lock every writer of a model's promotion log holds; refuses
    when it cannot be taken."""
    return _filelock.exclusive(
        _artifacts.run_dir(model_id) / PROMOTIONS_LOCK_FILE,
        required=True,
        purpose=f"the change to model {model_id!r}'s promotion log",
    )


def _parse(raw: bytes) -> Promotion:
    """One line as a record; raises ValueError, KeyError or TypeError."""
    record = json.loads(raw.decode("utf-8"))
    return Promotion(
        from_stage=str(record["from_stage"]),
        to_stage=str(record["to_stage"]),
        reason=str(record["reason"]),
        actor=str(record.get("actor", "unknown")),
        timestamp_utc=str(record["timestamp_utc"]),
        evidence=[str(e) for e in record.get("evidence") or []],
        prev=(None if record.get("prev") is None else str(record["prev"])),
    )


def _read_log(path: Path, model_id: str) -> Tuple[List[Promotion], Optional[bytes]]:
    """
    The committed records, and the torn fragment at the end if there is
    one: an unterminated final line that does not parse. An unterminated
    final line that DOES parse is a record whose newline was lost, and
    counts; any other unreadable line is an edit, and raises.
    """
    lines = path.read_bytes().split(b"\n")
    tail = lines.pop()  # whatever follows the last newline; b"" normally
    out: List[Promotion] = []
    for number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            out.append(_parse(raw))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValidationError(
                f"{path.name} line {number} for model {model_id!r} is not a "
                f"promotion record: {exc}. The log is append-only; a line that "
                "cannot be read means the file was edited, and the stage "
                "cannot be trusted until it is repaired."
            ) from exc
    if not tail.strip():
        return out, None
    try:
        out.append(_parse(tail))
    except (ValueError, KeyError, TypeError):
        return out, tail
    return out, None


def _repair_tail(path: Path) -> None:
    """
    Make the log end in a newline before anything is appended; the caller
    holds the lock.

    A final record that parses gets the newline it lost. A fragment that
    does not is copied to a `.promotions.torn-*` file beside the log, so
    the evidence of the interrupted write survives, and cut off.
    """
    if not path.exists():
        return
    with open(path, "rb+") as handle:
        data = handle.read()
        if not data or data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n") + 1
        tail = data[cut:]
        try:
            _parse(tail)
        except (ValueError, KeyError, TypeError):
            if tail.strip():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                evidence = path.with_name(
                    f"{_TORN_PREFIX}{stamp}-{uuid.uuid4().hex[:8]}"
                )
                evidence.write_bytes(tail)
                logger.warning(
                    "%s ended in an unterminated fragment of %d byte(s), left by "
                    "an interrupted write; it was not a record, and was moved "
                    "to %s",
                    path,
                    len(tail),
                    evidence.name,
                )
            handle.seek(cut)
            handle.truncate()
        else:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _line_bytes(record: Promotion) -> bytes:
    """The exact bytes one record is written as. The chain hashes THESE, so
    there is one definition of a line and the verifier cannot disagree with
    the writer about whitespace or key order."""
    return json.dumps(record.to_dict(), sort_keys=True).encode("utf-8") + b"\n"


def _chain_root(model_id: str) -> str:
    """The chain's root: the digest of the manifest bytes.

    The manifest is the package's commit point -- written last, carrying
    every other artifact's digest, and the one file the Ed25519 signature
    covers -- so a chain rooted there is rooted in the only thing about
    the package a signature already vouches for.
    """
    path = _artifacts.run_dir(model_id) / "manifest.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _committed_lines(path: Path) -> List[bytes]:
    """Every line the log has committed, by its newline, exactly as
    written. The torn fragment an interrupted append leaves is not one."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    parts = raw.split(b"\n")
    trailing = parts.pop()
    lines = [part + b"\n" for part in parts]
    if trailing:
        # An unterminated final line. It is a record if it parses -- a
        # newline lost rather than a write interrupted -- and `_read_log`
        # makes the same distinction.
        try:
            _parse(trailing)
        except (ValueError, KeyError, TypeError):
            return lines
        lines.append(trailing)
    return lines


def _chain_tip(path: Path, model_id: str) -> str:
    """What the next record's `prev` must be."""
    lines = _committed_lines(path)
    if not lines:
        return _chain_root(model_id)
    return hashlib.sha256(lines[-1]).hexdigest()


def verify_promotion_chain(model_id: str) -> List[str]:
    """
    One sentence per broken link, oldest first; empty for a log whose
    links all hold and for a log that has none.

    Reads the lines as bytes and recomputes each record's `prev` from the
    line before it, rooted in the manifest digest.

    See this module's docstring for what that does and does not catch. In
    short: an edit anywhere but the LAST record is detected, because the
    record after it commits to its bytes and nothing commits to the tip's;
    and a log rewritten end to end with every link recomputed is not
    detected at all. Both are structural, so they are documented rather
    than returned as findings -- this reports damage it can see, not the
    shape of what it cannot.
    """
    path = _log_path(model_id)
    lines = _committed_lines(path)
    findings: List[str] = []
    expected = _chain_root(model_id)
    for position, line in enumerate(lines):
        try:
            record = _parse(line)
        except (ValueError, KeyError, TypeError):
            findings.append(
                f"record {position + 1} of {len(lines)} does not parse, so the "
                "chain cannot be followed past it; every stage after it is "
                "unreadable."
            )
            return findings
        if record.prev is None:
            findings.append(
                f"record {position + 1} of {len(lines)} ({record.from_stage} -> "
                f"{record.to_stage}) carries no `prev`: it was written before "
                "the log was chained, so nothing vouches for it."
            )
        elif record.prev != expected:
            findings.append(
                f"record {position + 1} of {len(lines)} ({record.from_stage} -> "
                f"{record.to_stage}, {record.timestamp_utc}) links to "
                f"{record.prev[:12]}... and the line before it hashes to "
                f"{expected[:12]}...: a record was edited, removed or "
                "reordered at or before this point."
            )
        expected = hashlib.sha256(line).hexdigest()
    return findings


def _append(path: Path, record: Promotion) -> None:
    """One line, whole, and on disk before the lock is released."""
    line = _line_bytes(record)
    with open(path, "ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def promotions(model_id: str) -> List[Promotion]:
    """Every promotion recorded for the model, oldest first; empty for a
    model that was never promoted, which is every model at registration.

    A torn final line -- the fragment an interrupted append leaves -- is
    not a record and is skipped. When one is found it is also repaired,
    under the promotion lock: a fragment that is really an append still in
    progress is complete by the time the lock is held, and is then left
    alone. A reader that cannot take the lock skips the repair rather than
    fail.
    """
    path = _log_path(model_id)
    if not path.exists():
        return []
    records, torn = _read_log(path, model_id)
    if torn is not None:
        _repair_quietly(model_id, path)
    return records


def _repair_quietly(model_id: str, path: Path) -> None:
    """Repair a torn tail if the lock can be taken; a reader never fails
    because the repair could not be made."""
    handle = _filelock.acquire_lock(_artifacts.run_dir(model_id) / PROMOTIONS_LOCK_FILE)
    if handle is None:
        return
    try:
        _repair_tail(path)
    except OSError:
        logger.debug("[lifecycle] torn tail left in place", exc_info=True)
    finally:
        _filelock.release_lock(handle)


def torn_fragments(model_id: str) -> List[str]:
    """
    One sentence per fragment an interrupted append left at the end of the
    model's promotion log and a repair set aside, naming the side file it
    is kept in, oldest first; empty for a log that was never torn.

    Read from the side files, not from whichever call made the repair. Any
    reader repairs the log quietly (see `promotions`) -- `list_models` or
    `monitor_model` as readily as a promotion -- and a repair one of those
    made has to show on the model's card and at its next promotion all the
    same: the fragment may be the start of a decision somebody believes
    was recorded.
    """
    directory = _log_path(model_id).parent
    notes: List[str] = []
    for evidence in sorted(directory.glob(f"{_TORN_PREFIX}*")):
        try:
            size = evidence.stat().st_size
        except OSError:
            continue
        notes.append(
            f"{PROMOTIONS_FILE} ended in an unterminated fragment of {size} "
            "byte(s), left by an interrupted append; it was not a promotion "
            "record, so it was cut off the log and kept in "
            f"{evidence.name} beside it. If a decision was being recorded "
            "at the time, it is not in the history: record it again."
        )
    return notes


def current_stage(model_id: str) -> str:
    """The stage the last promotion moved the model to, or `candidate`."""
    history = promotions(model_id)
    return history[-1].to_stage if history else INITIAL_STAGE


def promote(
    model_id: str,
    to_stage: str,
    reason: str,
    *,
    actor: str = "agent",
    evidence: Sequence[str] = (),
) -> Promotion:
    """
    Record a stage change, after checking it is one the lifecycle allows.

    Refused by name: an unknown stage, a promotion to the stage the model is
    already at, any move out of `archived`, a forward move that skips a
    stage, and a reason too short to be one.

    Read, check and append happen under the model's promotion lock, so two
    concurrent promotions are decided one after the other against the
    stage each actually follows -- the second of two identical ones is
    refused as already made. Refused when the lock cannot be taken.
    """
    if to_stage not in STAGES:
        raise ValidationError(
            f"promote_model: {to_stage!r} is not a lifecycle stage; the stages "
            f"are {list(STAGES)}, in that order."
        )
    if not reason or len(reason.strip()) < 8:
        raise ValidationError(
            "promote_model: a promotion needs a reason of at least a few "
            "words. It is read months later by someone deciding whether to "
            "trust the model, and 'ok' does not help them."
        )
    path = _log_path(model_id)
    with promotions_lock(model_id):
        # Re-read INSIDE the lock: the stage checked is the stage appended
        # after, whoever else is promoting the same model.
        history = _read_log(path, model_id)[0] if path.exists() else []
        stage = history[-1].to_stage if history else INITIAL_STAGE
        record = _decide(model_id, stage, to_stage, reason, actor, evidence)
        # Inside the lock, like the stage: the tip is read and extended
        # without another writer landing a line between the two.
        record = replace(record, prev=_chain_tip(path, model_id))
        _repair_tail(path)
        _append(path, record)
        # The log is the stage; a mirror holding a stale log holds a stale
        # stage, so the whole file follows every decision -- inside the
        # lock, so the mirror receives the log in the order it was written.
        _mirror.mirror_file(model_id, PROMOTIONS_FILE)
    return record


def _decide(
    model_id: str,
    stage: str,
    to_stage: str,
    reason: str,
    actor: str,
    evidence: Sequence[str],
) -> Promotion:
    """The record for `stage -> to_stage`, or the refusal the rules give."""
    if stage == "archived":
        raise ValidationError(
            f"promote_model: model {model_id!r} is archived, which is "
            "terminal. Retrain to get a new model_id rather than reviving "
            "one whose history says it was retired."
        )
    if to_stage == stage:
        raise ValidationError(
            f"promote_model: model {model_id!r} is already at {stage!r}."
        )
    if to_stage != "archived" and _ORDER[to_stage] > _ORDER[stage] + 1:
        skipped = STAGES[_ORDER[stage] + 1 : _ORDER[to_stage]]
        raise ValidationError(
            f"promote_model: {stage!r} -> {to_stage!r} skips {list(skipped)}. "
            "A model reaches production one stage at a time, so every step "
            "is a decision somebody made and recorded; promote it to "
            f"{skipped[0]!r} first."
        )
    return Promotion(
        from_stage=stage,
        to_stage=to_stage,
        reason=reason.strip(),
        actor=str(actor or "agent"),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        evidence=[str(e) for e in evidence],
    )


__all__ = [
    "INITIAL_STAGE",
    "PROMOTIONS_FILE",
    "PROMOTIONS_LOCK_FILE",
    "STAGES",
    "Promotion",
    "current_stage",
    "promote",
    "promotions",
    "promotions_lock",
    "torn_fragments",
    "verify_promotion_chain",
]
