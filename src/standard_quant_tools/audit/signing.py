"""
Ed25519 checkpoint signing -- an optional external anchor on top of the
hash chain. `verify_audit_log_integrity`/`verify_audit_trail_integrity`
explicitly cannot catch an attacker who consistently rewrites an entire day
file *and* its chain-index entry to stay internally self-consistent (see
`verify.py`'s docstrings) -- there is no anchor outside those files
themselves. A signed checkpoint is that anchor: `verify_checkpoint_signature`
only needs the public key, not any trust in the JSONL files' own internal
consistency.

`cryptography` is an optional dependency
(`pip install standard_quant_tools[signing]`) -- every other part of the
audit trail works without it. Calling anything in this module without it
installed raises a clear `ImportError` with install instructions, the same
"graceful, explicit failure" contract `data.bloomberg_provider` uses for
`blpapi` and `_sqt_core`'s C++ extension uses elsewhere in this codebase.

Key custody is explicitly NOT this library's problem. `SQT_AUDIT_SIGNING_KEY_PATH`
(or an explicit `key_path`) points at a raw Ed25519 private key file for
local development -- `generate_keypair()` / `sqt keygen` make one, and are
labeled for that purpose only, not production key custody. A real
deployment should pass its own `signer: Callable[[bytes], bytes]` instead,
routed through an HSM/KMS, and never let this library see a bare private
key at all.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from standard_quant_tools._env import env_path
from standard_quant_tools.error import ValidationError

from .hashing import hash_payload
from .paths import _INDEX_FILENAME, _audit_dir
from .verify import _lines, _read_line, _walk_day

#: The checkpoint format `checkpoint_and_sign` writes. A checkpoint with no
#: `checkpoint_version` field is format 1 -- `{date, final_record_hash,
#: index_hash, signed_at_utc}` -- and is still verified by the rule it was
#: signed under.
CHECKPOINT_VERSION = 2

HAS_CRYPTOGRAPHY = False
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    HAS_CRYPTOGRAPHY = True
except ImportError:
    pass


def _require_cryptography() -> None:
    if not HAS_CRYPTOGRAPHY:
        raise ImportError(
            "cryptography is not installed. Ed25519 checkpoint signing "
            "requires it, and it isn't a hard dependency of this package. "
            "Install it with `pip install standard_quant_tools[signing]` "
            "(or `pip install cryptography` directly)."
        )


def generate_keypair() -> Tuple[bytes, bytes]:
    """
    Generate a new Ed25519 keypair, returned as `(private_bytes, public_bytes)`
    in raw encoding. For local development only — **not** a production key
    custody solution. A real deployment should generate/store keys through
    its own KMS/HSM and pass a `signer` callback to `checkpoint_and_sign`
    instead of ever writing a bare private key file.
    """
    _require_cryptography()
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private_bytes, public_bytes


def _load_signer(key_path: Optional[Union[str, Path]]) -> Callable[[bytes], bytes]:
    """
    Resolve a signing callback from a raw Ed25519 private key file: the
    explicit `key_path` param if given, else `SQT_AUDIT_SIGNING_KEY_PATH`.
    Raises a clear error if neither resolves to an existing file — the
    caller should pass their own `signer` callback instead if they don't
    want a bare key file on disk at all.

    The variable is read like every other path setting: blank is unset,
    `~` is expanded, and a relative path or one naming a directory is
    refused by name without the value being repeated -- a key's location
    is not something a refusal, which can end up in a log, should echo.
    """
    _require_cryptography()
    path = Path(key_path) if key_path else None
    if path is None:
        path = env_path("SQT_AUDIT_SIGNING_KEY_PATH", kind="file")
    if path is None or not path.exists():
        raise FileNotFoundError(
            "No signing key found. Pass key_path=..., set "
            "SQT_AUDIT_SIGNING_KEY_PATH, or pass your own `signer` callback "
            "(e.g. routed through an HSM/KMS) instead of a bare key file."
        )
    private_key = Ed25519PrivateKey.from_private_bytes(path.read_bytes())

    def _sign(payload: bytes) -> bytes:
        return private_key.sign(payload)

    return _sign


@dataclass(frozen=True)
class _RecomputedDay:
    """A day as its records recompute NOW, from the chain index's head.

    The checkpoint used to be compared with the `record_hash` the day's
    last line CLAIMS. Nothing recomputed it, so an edit that left the
    stored hashes alone changed nothing the checkpoint looked at: a day
    whose records had been edited reported "valid", and `sqt verify
    --checkpoint` printed "Signature valid." with exit 0. Recomputing is
    what makes the signature a statement about the records rather than
    about one string on the last line. See the CHANGELOG entry of
    2026-09-27.
    """

    #: record_hash of each record in the leading run whose link and content
    #: both hold, in file order.
    clean_hashes: List[str]
    #: Those records themselves, parsed, in the same order.
    clean_records: List[Dict[str, Any]]
    #: Non-blank lines in the day file, readable or not.
    lines: int
    #: Every line holds, from the index's head to the end of the file.
    whole_day_holds: bool
    #: The first line where the recomputed chain stops holding, if any.
    first_break: Optional[int]
    #: The chain index's entry for this date (the last one, if repeated).
    index_entry: Optional[Dict[str, Any]]
    #: Whether that entry's own hash recomputes to what it claims.
    index_entry_holds: bool


def _index_entries(directory: Path) -> List[Optional[Dict[str, Any]]]:
    """Every non-blank line of the chain index, in order: the entry, or None
    for a line that is not one (the trail check's finding to report)."""
    index_path = directory / _INDEX_FILENAME
    if not index_path.exists():
        return []
    return [_read_line(raw)[0] for _, raw in _lines(index_path)]


def _index_entry(date: str, directory: Path) -> Optional[Dict[str, Any]]:
    """The chain index's entry for `date` -- the last one, since the index is
    append-only and the most recent claim is what the writer committed to.
    An unreadable index line names no date and is the trail check's finding
    to report."""
    found: Optional[Dict[str, Any]] = None
    for entry in _index_entries(directory):
        if entry is not None and entry.get("date") == date:
            found = entry
    return found


def _recompute_day(date: str, directory: Path) -> _RecomputedDay:
    """Walk `date` from the head the chain index recorded for it, through the
    same per-record checks `verify_audit_log_integrity` makes. A day the
    index never witnessed is walked from its first record's own claim. A
    line that is not a record stops the day holding there."""
    entry = _index_entry(date, directory)
    entry_holds = entry is None or (
        hash_payload({**entry, "index_hash": None}) == entry.get("index_hash")
    )
    head = entry.get("chain_head") if entry is not None else None

    day_path = directory / f"{date}.jsonl"
    clean: List[str] = []
    records: List[Dict[str, Any]] = []
    holding = True
    first_break: Optional[int] = None
    lines = 0
    if day_path.exists():
        for check in _walk_day(day_path, head):
            lines += 1
            if (
                holding
                and check.unreadable is None
                and check.link_holds
                and check.content_holds
            ):
                clean.append(check.claimed_hash)
                records.append(check.record)
            elif holding:
                holding = False
                first_break = check.lineno
    return _RecomputedDay(
        clean_hashes=clean,
        clean_records=records,
        lines=lines,
        whole_day_holds=holding and len(clean) == lines,
        first_break=first_break,
        index_entry=entry,
        index_entry_holds=entry_holds,
    )


def _records_digest(records: List[Dict[str, Any]]) -> str:
    """
    A full SHA-256 over records, each as canonical JSON (sorted keys), one
    per line.

    The record chain itself is 64-bit: each `record_hash` is SHA-256 cut to
    16 hex characters, and it stays that way because every record already
    written carries one and re-hashing history is indistinguishable from
    rewriting it. A signature over one 64-bit endpoint committed to 64 bits
    of the day. This digest is what a checkpoint signs instead, so a signed
    day is bound at 256 bits with no change to a single record. It is taken
    over the parsed records rather than the file's bytes, so a copy whose
    line endings were converted still verifies.
    """
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def checkpoint_and_sign(
    date: str,
    audit_dir: Optional[Union[str, Path]] = None,
    key_path: Optional[Union[str, Path]] = None,
    signer: Optional[Callable[[bytes], bytes]] = None,
) -> Path:
    """
    Build a checkpoint for `date` and sign it with Ed25519, writing
    `<date>.checkpoint.json` (the checkpoint payload) and
    `<date>.checkpoint.sig` (the raw signature, hex-encoded) as sidecars in
    the audit directory. A periodic signed checkpoint, not a signature on
    every record, is enough: the hash chain already covers per-record
    integrity, the checkpoint anchors the chain's endpoint for that day.

    The checkpoint is format 2 (`checkpoint_version: 2`): `date`,
    `record_count`, `final_record_hash`, `day_digest` (a full SHA-256 over
    the day's records -- see `_records_digest`), `index_hash` (the day's
    chain-index entry), `index_len` and `index_head` (how many lines the
    chain index held when the day was signed, and the last one's hash), and
    `signed_at_utc`. Format 1 signed only the day's last 64-bit
    record_hash and its index entry, so a truncated index or a second
    preimage of one short hash was all it took; format 2 anchors the
    record count at 256 bits and the index's length up to the moment of
    signing. See the CHANGELOG entry of 2026-09-28.

    The endpoint signed is the one the day's records RECOMPUTE to, walked
    from the chain index's head. A day whose chain does not hold is refused
    with a ValidationError rather than signed: a signature over a damaged
    day would certify the damage, and every later verification would then
    compare against it. A day with no records is refused too -- there is
    nothing to anchor.

    Signing key: pass `signer` (e.g. routed through an HSM/KMS) for
    anything beyond local development, OR `key_path` / the
    `SQT_AUDIT_SIGNING_KEY_PATH` env var pointing at a raw Ed25519 private
    key file (see `generate_keypair`/`sqt keygen` — development only).

    Returns the checkpoint JSON path.
    """
    _require_cryptography()
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    day = _recompute_day(date, directory)
    if day.lines == 0:
        raise ValidationError(
            f"{date} has no records in {directory}, so there is nothing to "
            "anchor. Sign a day that has activity; describe_audit_log or the "
            "day files' names say which days do."
        )
    if not day.index_entry_holds:
        raise ValidationError(
            f"the chain index entry for {date} does not match its own "
            "index_hash, so the head this day chains from cannot be trusted "
            "and the day is not signed. Run `sqt verify` to see what broke."
        )
    if not day.whole_day_holds:
        raise ValidationError(
            f"{date}'s records do not recompute to an intact chain (it stops "
            f"holding at line {day.first_break}), so the day is not signed: "
            "a signature over a damaged day would certify the damage. Run "
            "`sqt verify` to see what broke, and restore the day from an "
            "exported bundle or a backup before anchoring it."
        )
    index_lines = _index_entries(directory)
    index_last = index_lines[-1] if index_lines else None
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "date": date,
        "record_count": len(day.clean_hashes),
        "final_record_hash": day.clean_hashes[-1],
        "day_digest": _records_digest(day.clean_records),
        "index_hash": (
            day.index_entry.get("index_hash") if day.index_entry is not None else None
        ),
        "index_len": len(index_lines),
        "index_head": (
            index_last.get("index_hash") if index_last is not None else None
        ),
        "signed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    canonical = json.dumps(checkpoint, sort_keys=True).encode("utf-8")
    sign_fn = signer if signer is not None else _load_signer(key_path)
    signature = sign_fn(canonical)

    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"{date}.checkpoint.json"
    sig_path = directory / f"{date}.checkpoint.sig"
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8"
    )
    sig_path.write_text(signature.hex(), encoding="utf-8")
    return checkpoint_path


# Raw Ed25519 signature length, in bytes. A .sig file that does not decode
# to exactly this is damaged rather than merely wrong.
_ED25519_SIGNATURE_BYTES = 64

#: Every state `verify_checkpoint_state` can return, in the order a reader
#: meets them.
CHECKPOINT_STATES: Tuple[str, ...] = (
    "valid",
    "extended",
    "altered",
    "no_checkpoint",
    "no_signature",
    "key_mismatch",
    "corrupt_signature",
    "unavailable",
)

#: The states where a check RAN and did not pass. The others are the
#: absence of evidence (a day nobody anchored, a checkpoint with no
#: signature, a check that could not be made) or, for `extended`, a day
#: that has only grown since it was signed.
CHECKPOINT_FAILURES = frozenset({"altered", "key_mismatch", "corrupt_signature"})

#: What each state means and what to do about it, shared by every surface
#: that reports one (the `sqt` CLI and the meta runtime's
#: `verify_audit_integrity`), so they cannot describe a state two ways.
CHECKPOINT_STATE_NOTES: Dict[str, str] = {
    "valid": (
        "The signature verifies under this public key, and the day's "
        "records, recomputed from the chain index's head, end exactly where "
        "the signed checkpoint says."
    ),
    "extended": (
        "The signature verifies and every record it covers still recomputes "
        "to the endpoint it signed; more records were appended after "
        "signing. Those later records are covered by the hash chain only. "
        "This is what a day still being written to looks like: re-anchor "
        "the day once it has closed."
    ),
    "altered": (
        "The signature verifies, but the day's records no longer recompute "
        "to the endpoint it signed: a record was edited, removed or cut off, "
        "the day was rewritten, or its chain-index entry changed. Treat the "
        "day as tampered and compare it with an exported bundle or a backup."
    ),
    "no_checkpoint": (
        "No checkpoint file exists for this day, so there is nothing to "
        "verify -- the day was never anchored. This is not evidence of "
        "tampering; it is the absence of the evidence that would detect it."
    ),
    "no_signature": (
        "A checkpoint exists for this day but its signature file does not, "
        "so the checkpoint is unsigned and proves nothing on its own."
    ),
    "key_mismatch": (
        "The signature is well-formed but was not made by the key that "
        "matches the public key supplied. Either the wrong public key was "
        "given, or the checkpoint was signed by someone else."
    ),
    "corrupt_signature": (
        "The signature file could not be read as a signature over this "
        "checkpoint -- truncated, re-encoded or altered bytes."
    ),
    "unavailable": (
        "The signature could not be checked at all -- no public key file at "
        "that path, an unreadable key or checkpoint, a day file that cannot "
        "be read, or no Ed25519 implementation installed. This is a MISSING "
        "check, not a failed one, so it is reported as unknown rather than "
        "as a broken signature."
    ),
}


@dataclass(frozen=True)
class CheckpointVerification:
    """What `verify_checkpoint` found for one day.

    `state` is one of `CHECKPOINT_STATES`. `records_signed` is how many of
    the day's records the signature covers and `records_after` how many
    were appended after them -- 0 for "valid", at least 1 for "extended",
    None wherever the signed endpoint could not be located. `detail` says,
    in one sentence, where an "altered" day stops holding or how far an
    "extended" one has grown. `version` is the checkpoint's format (1 or
    2; None when no checkpoint was read): format 2 also commits to the
    record count, a full SHA-256 digest of the records and the chain
    index's length.
    """

    state: str
    records_signed: Optional[int] = None
    records_after: Optional[int] = None
    detail: Optional[str] = None
    version: Optional[int] = None


def _content_state(
    date: str, stored_checkpoint: Dict[str, Any], directory: Path
) -> CheckpointVerification:
    """What was signed, compared with the day as it recomputes now, tagged
    with the checkpoint's format."""
    version = stored_checkpoint.get("checkpoint_version", 1)
    if version not in (1, 2) or isinstance(version, bool):
        return CheckpointVerification(
            "unavailable",
            detail=(
                f"the checkpoint says it is format {version!r}, which this "
                "release does not read; verify it with the release that "
                "signed it"
            ),
        )
    found = _compare_with_day(date, stored_checkpoint, directory, version)
    return CheckpointVerification(
        found.state,
        records_signed=found.records_signed,
        records_after=found.records_after,
        detail=found.detail,
        version=version,
    )


def _index_anchor_state(
    stored_checkpoint: Dict[str, Any], directory: Path
) -> Optional[CheckpointVerification]:
    """A format-2 checkpoint's hold on the chain index: the index must still
    hold at least as many lines as when the day was signed, and the line
    that was last then must be the same entry. None when it holds.

    Deleting the newest day together with the index's last line, or
    trimming the index back, leaves a shorter trail that verifies clean on
    its own. A signature over the index's length at signing is what makes
    that visible for every entry up to the signed day."""
    signed_len = stored_checkpoint.get("index_len")
    if isinstance(signed_len, bool) or not isinstance(signed_len, int):
        return CheckpointVerification(
            "altered", detail="the checkpoint carries no chain index length"
        )
    lines = _index_entries(directory)
    if len(lines) < signed_len:
        return CheckpointVerification(
            "altered",
            detail=(
                f"the chain index holds {len(lines)} line(s), fewer than the "
                f"{signed_len} it held when this day was signed: entries were "
                "removed"
            ),
        )
    if signed_len:
        line = lines[signed_len - 1]
        found = line.get("index_hash") if line is not None else None
        if found != stored_checkpoint.get("index_head"):
            return CheckpointVerification(
                "altered",
                detail=(
                    f"chain index line {signed_len} is not the entry that was "
                    "last when this day was signed: the index was rewritten"
                ),
            )
    return None


def _compare_with_day(
    date: str, stored_checkpoint: Dict[str, Any], directory: Path, version: int
) -> CheckpointVerification:
    """Compare what was signed with the day as it recomputes now.

    valid     the signed endpoint is the last record, and every record from
              the index's head to it holds
    extended  the signed endpoint is inside an intact day and every record
              after it is an intact continuation
    altered   anything else: the signed endpoint is not reachable through
              records that recompute (an edit, a truncation, a rewrite), a
              record after it does not hold, or the index entry changed

    Format 1 finds the signed endpoint by its 64-bit record_hash, exactly
    as it always has, so every checkpoint already signed is judged by the
    rule it was signed under and an untouched day still reads "valid".
    Format 2 also requires the endpoint to be record `record_count`, the
    records up to it to match `day_digest` at 256 bits, and the chain index
    to hold what it held at signing.
    """
    day = _recompute_day(date, directory)
    entry = day.index_entry
    current_index_hash = entry.get("index_hash") if entry is not None else None
    if stored_checkpoint.get("index_hash") != current_index_hash:
        return CheckpointVerification(
            "altered",
            detail=(
                "the chain index entry for this day is not the one that was " "signed"
            ),
        )
    if not day.index_entry_holds:
        return CheckpointVerification(
            "altered",
            detail=(
                "the chain index entry for this day no longer matches its "
                "own index_hash"
            ),
        )

    if version == 2:
        anchored = _index_anchor_state(stored_checkpoint, directory)
        if anchored is not None:
            return anchored

    signed = stored_checkpoint.get("final_record_hash")
    covered: Optional[int]
    if version == 2:
        count = stored_checkpoint.get("record_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            return CheckpointVerification(
                "altered", detail="the checkpoint carries no usable record count"
            )
        if len(day.clean_hashes) < count:
            where = (
                f", and the recomputed chain stops holding at line "
                f"{day.first_break}"
                if day.first_break is not None
                else ""
            )
            return CheckpointVerification(
                "altered",
                detail=(
                    f"{count} record(s) were signed and only "
                    f"{len(day.clean_hashes)} still recompute{where}: the day "
                    "was cut short, edited or rewritten"
                ),
            )
        if day.clean_hashes[count - 1] != signed:
            return CheckpointVerification(
                "altered",
                detail=(
                    f"record {count} is not the signed endpoint {signed!r}: the "
                    "day was rewritten"
                ),
            )
        if _records_digest(day.clean_records[:count]) != stored_checkpoint.get(
            "day_digest"
        ):
            return CheckpointVerification(
                "altered",
                detail=(
                    f"the {count} signed record(s) no longer match the signed "
                    "SHA-256 digest of the day, although their 64-bit chain "
                    "still links: a record was replaced by one with a "
                    "colliding record_hash"
                ),
            )
        covered = count
    elif signed is None:
        # Signed while the day had no records (possible before signing
        # refused an empty day): every record now present came after it.
        covered = 0
    elif signed in day.clean_hashes:
        covered = day.clean_hashes.index(signed) + 1
    else:
        covered = None

    if covered is None:
        where = (
            f"; the recomputed chain stops holding at line {day.first_break}"
            if day.first_break is not None
            else "; the day was cut short or rewritten"
        )
        return CheckpointVerification(
            "altered",
            detail=f"the signed endpoint {signed!r} is not reached by records "
            f"that recompute{where}",
        )
    if not day.whole_day_holds:
        return CheckpointVerification(
            "altered",
            records_signed=covered,
            detail=(
                f"the {covered} signed record(s) still hold, but the chain "
                f"stops holding at line {day.first_break}, after them"
            ),
        )
    after = day.lines - covered
    if after == 0:
        return CheckpointVerification("valid", records_signed=covered, records_after=0)
    return CheckpointVerification(
        "extended",
        records_signed=covered,
        records_after=after,
        detail=(
            f"{after} record(s) were appended after the {covered} the "
            "signature covers"
        ),
    )


def verify_checkpoint(
    date: str,
    public_key_path: Union[str, Path],
    audit_dir: Optional[Union[str, Path]] = None,
) -> CheckpointVerification:
    """
    Verify a day's signed checkpoint: the signature, with only the public
    key, and then what it signed against the day's records RECOMPUTED from
    the chain index's head -- not against the hash the last line claims,
    which an edit that leaves the stored hashes alone does not change.

    Returns a `CheckpointVerification` whose `state` names which of
    `CHECKPOINT_STATES` the day is in (see the CHANGELOG entry of
    2026-09-22 for why a single bool was not enough, and of 2026-09-27 for
    why `extended` and `altered` are separate):

        "valid"             the signature checks out and the day ends
                            exactly where it was signed
        "extended"          the signature checks out, the records it covers
                            still hold, and more were appended after them --
                            ordinary for a day still being written to
        "altered"           the signature checks out but the day no longer
                            recomputes to what it signed: an edit, a
                            truncation, a rewrite, a changed index entry
        "no_checkpoint"     this day was never anchored
        "no_signature"      a checkpoint exists with no .sig beside it
        "key_mismatch"      a well-formed signature that does not verify
                            under this public key — the wrong key, or
                            signature bytes altered in a way that is
                            indistinguishable from the wrong key
        "corrupt_signature" the .sig file does not decode to a 64-byte
                            Ed25519 signature at all
        "unavailable"       the check could not be made: no public key file,
                            an unreadable key, checkpoint or day file

    Order matters, and the signature is checked BEFORE the content: a
    later record never breaks a signature (it is taken over the stored
    checkpoint file), so a signature that fails is reported as a signature
    failure and never as a change to the day.

    Never raises for any of the above; `_require_cryptography()` still
    raises when `cryptography` itself is missing, since that is a statement
    about this installation rather than about the trail.
    """
    _require_cryptography()
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    checkpoint_path = directory / f"{date}.checkpoint.json"
    sig_path = directory / f"{date}.checkpoint.sig"
    if not checkpoint_path.exists():
        return CheckpointVerification("no_checkpoint")
    if not sig_path.exists():
        return CheckpointVerification("no_signature")

    try:
        stored_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        canonical = json.dumps(stored_checkpoint, sort_keys=True).encode("utf-8")
    except Exception:
        return CheckpointVerification("unavailable")
    if not isinstance(stored_checkpoint, dict):
        return CheckpointVerification("unavailable")

    try:
        signature = bytes.fromhex(sig_path.read_text(encoding="utf-8").strip())
    except Exception:
        return CheckpointVerification("corrupt_signature")
    if len(signature) != _ED25519_SIGNATURE_BYTES:
        return CheckpointVerification("corrupt_signature")

    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            Path(public_key_path).read_bytes()
        )
    except Exception:
        return CheckpointVerification("unavailable")

    try:
        public_key.verify(signature, canonical)  # raises InvalidSignature on mismatch
    except Exception:
        return CheckpointVerification("key_mismatch")

    try:
        return _content_state(date, stored_checkpoint, directory)
    except OSError:
        return CheckpointVerification("unavailable")


def verify_checkpoint_state(
    date: str,
    public_key_path: Union[str, Path],
    audit_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    The state name `verify_checkpoint` found -- one of `CHECKPOINT_STATES`,
    rather than one bool, because "never signed", "wrong key", "grown since
    signing" and "edited since signing" call for completely different
    responses. Never raises for a bad input.
    """
    return verify_checkpoint(date, public_key_path, audit_dir=audit_dir).state


def verify_checkpoint_signature(
    date: str,
    public_key_path: Union[str, Path],
    audit_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Verify a checkpoint's signature using **only the public key** —
    independent of trusting the JSONL files' own internal consistency — and
    confirm the day's records, recomputed from the chain index's head, still
    end exactly where the checkpoint says. A day whose records were edited
    after signing fails, whether or not the editor left the stored hashes
    alone; so does a day cut short.

    Returns `True` only for the "valid" state and `False` (never raises)
    for every other: a missing checkpoint or signature, a public key that
    doesn't match the signing key, a corrupted signature, a day altered
    since signing -- and a day that has only grown since signing
    ("extended"), since the records appended after signing are not covered
    by it. `verify_checkpoint_state` names which; this is that answer
    narrowed to the one bit a caller who only wants a gate needs.
    """
    return (
        verify_checkpoint_state(date, public_key_path, audit_dir=audit_dir) == "valid"
    )
