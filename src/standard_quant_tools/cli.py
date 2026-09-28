"""
Command-line interface for the audit trail's JSONL decision records
(the `standard_quant_tools.audit` package). Subcommands:

    sqt replay <request_id>              — re-run the recorded call via
                                            audit.verify_replay(), report
                                            whether data/output still match.
                                            Exit code: 0 = output matched,
                                            1 = output_match is False (a
                                            confirmed mismatch), 2 = the
                                            record has no output_hash to
                                            compare (indeterminate).
    sqt compare <request_id_a> <id_b>    — diff two records' status/output/
                                            timing/provenance and inputs.
    sqt report <request_id>              — pretty-print one record in full.
    sqt verify [--file PATH]             — check hash-chain integrity. With
                                            no args, verifies the full
                                            cross-day trail (every day file
                                            plus the chain index) via
                                            audit.verify_audit_trail_integrity().
                                            With --file, verifies just that
                                            one day file in isolation via
                                            audit.verify_audit_log_integrity().
                                            Exit code: 0 = clean, 1 = one or
                                            more problems found (printed to
                                            stdout, one per line). The last
                                            line names the head the check
                                            ran through, to be recorded
                                            somewhere else.
    sqt hold <date> [--reason TEXT]      — place a legal/retention hold on
                                            a calendar day (YYYY-MM-DD),
                                            protecting it from `sqt gc`.
    sqt release-hold <date>              — remove a hold from a day.
    sqt gc [--confirm]                   — delete day files past
                                            SQT_AUDIT_RETENTION_DAYS,
                                            excluding held days. Dry-run
                                            (lists candidates only) unless
                                            --confirm is passed. A negative
                                            window is refused.
    sqt runs gc [--confirm]
                [--older-than HOURS]     — list what interrupted writes
                                            left in SQT_RUNS_DIR (temp
                                            files; model/dataset directories
                                            with no commit file), older than
                                            24 hours by default. Dry-run
                                            unless --confirm. Published
                                            values are never collected.
    sqt seal <date>                      — chmod a day file read-only
                                            (not WORM — see
                                            audit.seal_day's docstring).
    sqt export --start D --end D --out F — package day files in [start,
                                            end] plus the chain index, a
                                            manifest, and the standalone
                                            verifier into one zip bundle.
                                            An existing file at F is
                                            refused, never replaced.
    sqt keygen [--out DIR]                — generate an Ed25519 signing
                                            keypair. Local development only
                                            — not production key custody.
    sqt anchor <date> [--key PATH]       — sign a checkpoint for a calendar
                                            day (see audit.checkpoint_and_sign).
                                            Key from --key or
                                            SQT_AUDIT_SIGNING_KEY_PATH.
    sqt verify --checkpoint <date>
               --pubkey PATH             — verify the full trail's hash chain
                                            AND that day's Ed25519 checkpoint
                                            (public key only), and print the
                                            checkpoint's state by name.
                                            Exit code: 0 = chain clean and
                                            checkpoint valid, 1 = a chain
                                            problem or a failed checkpoint
                                            (altered, key_mismatch,
                                            corrupt_signature), 2 = nothing
                                            to check or unable to check
                                            (no_checkpoint, no_signature,
                                            unavailable), 3 = extended (the
                                            day grew after it was signed).

`keygen`/`anchor`/`--checkpoint` verification require the optional
`cryptography` dependency (`pip install standard_quant_tools[signing]`) —
every other subcommand works without it. stdlib argparse only otherwise, no
new dependency, matching this repo's minimal-dependency stance. Each
subcommand's logic lives in its own `cmd_*` function so it can be tested
directly without spawning a subprocess.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from standard_quant_tools import audit


def _iter_records(audit_dir: Optional[Path] = None) -> Iterator[Dict[str, Any]]:
    """Decision records only -- excludes the chain-index witness log
    (_chain_index.jsonl, see the audit package's paths module), which lives
    in the same directory
    and matches the same *.jsonl glob but holds index entries, not
    decision records. A line that is not a record is skipped: finding one
    record must not fail on another's damage, which is `sqt verify`'s to
    report."""
    directory = audit_dir if audit_dir is not None else audit._audit_dir()
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.jsonl")):
        if not audit._DAY_FILE_RE.match(path.name):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record


def find_record(request_id: str, audit_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Find a decision record by request_id across every daily JSONL file in
    audit_dir (default: SQT_AUDIT_DIR / the package default).

    Raises:
        ValueError: no record with that request_id exists.
    """
    for record in _iter_records(audit_dir):
        if record.get("request_id") == request_id:
            return record
    raise ValueError(f"No decision record found for request_id={request_id!r}")


def cmd_report(request_id: str, audit_dir: Optional[Path] = None) -> str:
    """Pretty-printed JSON of one record's full fields."""
    record = find_record(request_id, audit_dir)
    return json.dumps(record, indent=2, sort_keys=True)


def _format_replay(result: audit.ReplayResult) -> str:
    lines = [
        f"request_id   : {result.request_id}",
        f"tool_name    : {result.tool_name}",
        f"output_match : {result.output_match}",
    ]
    for m in result.data_source_matches:
        lines.append(
            f"  data_source: {m['symbol']} {m['start']} -> {m['end']} "
            f"({m['interval']})  match={m['match']}"
        )
    for note in result.notes:
        lines.append(f"  note       : {note}")
    return "\n".join(lines)


def _replay_exit_code(result: audit.ReplayResult) -> int:
    """
    0 = output reproduced exactly, 1 = output_match is False (confirmed
    mismatch — code or data changed the result), 2 = output_match is None
    (the stored record has no output_hash to compare against, so replay
    success can't be determined either way).
    """
    if result.output_match is False:
        return 1
    if result.output_match is None:
        return 2
    return 0


def _replay(request_id: str, audit_dir: Optional[Path] = None) -> Tuple[str, int]:
    """Re-run a recorded call: the formatted report and the exit code.

    Both, from one place. `main` used to repeat these three lines inline
    because `cmd_replay` returns only the text and the CLI also needs the
    code -- so the function with the test and the code that actually ran
    were two copies, and a change to either would have left the test
    passing on a path the CLI does not take.
    """
    record = find_record(request_id, audit_dir)
    result = audit.verify_replay(record)
    return _format_replay(result), _replay_exit_code(result)


def cmd_replay(request_id: str, audit_dir: Optional[Path] = None) -> str:
    """Re-run the recorded call via audit.verify_replay() and format the result."""
    return _replay(request_id, audit_dir)[0]


def cmd_compare(
    request_id_a: str, request_id_b: str, audit_dir: Optional[Path] = None
) -> str:
    """Human-readable diff of two records' status/output/provenance/inputs."""
    a = find_record(request_id_a, audit_dir)
    b = find_record(request_id_b, audit_dir)

    lines = [f"Comparing {request_id_a} vs {request_id_b}", ""]
    fields = [
        "tool_name",
        "status",
        "output_hash",
        "duration_ms",
        "git_commit_sha",
        "package_version",
        "strategy_source_hash",
        "random_seed",
    ]
    for field in fields:
        va, vb = a.get(field), b.get(field)
        marker = "==" if va == vb else "!="
        lines.append(f"{field:22s} {marker}  {va!r}  vs  {vb!r}")

    input_a, input_b = a.get("input", {}), b.get("input", {})
    all_keys = sorted(set(input_a) | set(input_b))
    diffs = [k for k in all_keys if input_a.get(k) != input_b.get(k)]
    lines.append("")
    if diffs:
        lines.append("input differences:")
        for k in diffs:
            lines.append(f"  {k}: {input_a.get(k)!r}  vs  {input_b.get(k)!r}")
    else:
        lines.append("input: identical")

    return "\n".join(lines)


def cmd_verify(
    file: Optional[Path] = None,
    audit_dir: Optional[Path] = None,
    notes: Optional[List[str]] = None,
    head: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Check hash-chain integrity. `file` (a single day's .jsonl) checks just
    that file in isolation; no `file` checks the full cross-day trail
    (every day file plus the chain index) rooted at `audit_dir`.

    Returns a list of human-readable problems (empty if clean). `notes`,
    when given, receives what was found and is not a problem; `head`, when
    given, what the check ran through (see `audit.describe_head`).
    """
    if file is not None:
        return audit.verify_audit_log_integrity(file, notes=notes, head=head)
    return audit.verify_audit_trail_integrity(audit_dir, notes=notes, head=head)


def _format_verify(
    problems: List[str],
    notes: Optional[List[str]] = None,
    head: Optional[Dict[str, Any]] = None,
) -> str:
    if not problems:
        lines = ["OK — no integrity problems found."]
    else:
        lines = [f"{len(problems)} problem(s) found:"]
        lines.extend(f"  - {p}" for p in problems)
    if notes:
        lines.append(f"{len(notes)} note(s), not problems:")
        lines.extend(f"  - {n}" for n in notes)
    if head is not None:
        # The last line names where the check ran through, so it can be
        # recorded somewhere this directory cannot reach: a newest day cut
        # short verifies clean against the files alone.
        lines.append(audit.describe_head(head))
    return "\n".join(lines)


def cmd_hold(
    date: str, reason: Optional[str] = None, audit_dir: Optional[Path] = None
) -> Path:
    return audit.hold_day(date, audit_dir=audit_dir, reason=reason)


def cmd_release_hold(date: str, audit_dir: Optional[Path] = None) -> bool:
    return audit.release_hold(date, audit_dir=audit_dir)


def cmd_gc(
    confirm: bool = False,
    retention_days: Optional[int] = None,
    audit_dir: Optional[Path] = None,
) -> List[str]:
    """Dry-run (confirm=False, the default): returns candidate dates without
    deleting anything. confirm=True: actually deletes and returns the dates
    that were deleted."""
    return audit.gc(
        audit_dir=audit_dir, retention_days=retention_days, dry_run=not confirm
    )


def cmd_cache_gc(confirm: bool = False) -> List[Path]:
    """Dry-run (the default) lists the OHLCV cache files of a dead format
    generation, then the temp files a cache write left behind and nobody
    can still own (older than an hour); confirm=True deletes them. Nothing
    else is evicted: the current generation is the cache, and files without
    a generation prefix are not this cache's to remove."""
    dead, orphans = _cache_gc(confirm)
    return dead + orphans


def _cache_gc(confirm: bool) -> "tuple[List[Path], List[Path]]":
    from standard_quant_tools.data._cache import dead_generations, orphaned_temps

    return dead_generations(dry_run=not confirm), orphaned_temps(dry_run=not confirm)


def cmd_seal(date: str, audit_dir: Optional[Path] = None) -> Path:
    return audit.seal_day(date, audit_dir=audit_dir)


def cmd_runs_gc(confirm: bool = False, older_than_hours: Optional[float] = None):
    """Dry-run (the default) lists what interrupted writes left in the runs
    directory -- temp files, and model or dataset directories whose
    registration never committed -- older than the threshold; confirm=True
    deletes them. A published value is never a candidate: deleting one
    would break every reference to it (`_runspath.sweep`)."""
    from standard_quant_tools._runspath import DEFAULT_SWEEP_HOURS, sweep

    hours = DEFAULT_SWEEP_HOURS if older_than_hours is None else older_than_hours
    return sweep(older_than_hours=hours, dry_run=not confirm)


def _print_runs_gc(report, confirm: bool) -> None:
    if not report.candidates:
        print(
            f"Nothing to collect in {report.root} older than "
            f"{report.older_than_hours:g} hour(s)."
        )
    verb = "Deleted" if confirm else "Would delete (dry-run; pass --confirm)"
    for label, paths in (
        ("temp file(s) left by interrupted writes", report.temp_files),
        ("model/dataset director(ies) never registered", report.partial_directories),
    ):
        if paths:
            print(f"{verb}: {len(paths)} {label}")
            for path in paths:
                print(f"  - {path.relative_to(report.root)}")
    print(
        "Published values are never collected: deleting one breaks every "
        "reference to it."
    )


def cmd_export(
    start: str, end: str, out: Path, audit_dir: Optional[Path] = None
) -> Path:
    return audit.export_bundle(start, end, out, audit_dir=audit_dir)


def cmd_keygen(out_dir: Path) -> "tuple[Path, Path]":
    """Generate an Ed25519 keypair and write it as two files
    (audit_signing_key.private / .public) under out_dir. Local development
    only -- see audit.generate_keypair's docstring."""
    private_bytes, public_bytes = audit.generate_keypair()
    out_dir.mkdir(parents=True, exist_ok=True)
    priv_path = out_dir / "audit_signing_key.private"
    pub_path = out_dir / "audit_signing_key.public"
    priv_path.write_bytes(private_bytes)
    pub_path.write_bytes(public_bytes)
    return priv_path, pub_path


def cmd_anchor(
    date: str, key_path: Optional[Path] = None, audit_dir: Optional[Path] = None
) -> Path:
    return audit.checkpoint_and_sign(date, audit_dir=audit_dir, key_path=key_path)


def cmd_verify_checkpoint(
    date: str, pubkey: Path, audit_dir: Optional[Path] = None
) -> "audit.CheckpointVerification":
    """The checkpoint's full state, not a boolean. A bool printed "Signature
    invalid" for a day nobody signed, a missing key file and a record
    appended after signing alike, and each calls for a different response.
    """
    return audit.verify_checkpoint(date, pubkey, audit_dir=audit_dir)


#: Exit code per checkpoint state. 0 only for "valid"; 1 where a check ran
#: and failed, as for a chain problem; 2 where there was nothing to check or
#: no way to check it; 3 for a day that has only grown since it was signed,
#: which a script watching a day still being written to must be able to tell
#: apart from both.
_CHECKPOINT_EXIT_CODES = {
    "valid": 0,
    "altered": 1,
    "key_mismatch": 1,
    "corrupt_signature": 1,
    "no_checkpoint": 2,
    "no_signature": 2,
    "unavailable": 2,
    "extended": 3,
}


def _format_checkpoint(date: str, found: "audit.CheckpointVerification") -> str:
    if found.state == "valid":
        return f"Checkpoint {date}: valid\nSignature valid."
    from standard_quant_tools.audit.signing import CHECKPOINT_STATE_NOTES

    lines = [f"Checkpoint {date}: {found.state}"]
    note = CHECKPOINT_STATE_NOTES.get(found.state)
    if note:
        lines.append(f"  {note}")
    if found.detail:
        lines.append(f"  ({found.detail})")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sqt", description="standard_quant_tools audit-trail CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_replay = sub.add_parser(
        "replay", help="Re-run a recorded tool call and check data/output match."
    )
    p_replay.add_argument("request_id")

    p_compare = sub.add_parser("compare", help="Diff two recorded tool calls.")
    p_compare.add_argument("request_id_a")
    p_compare.add_argument("request_id_b")

    p_report = sub.add_parser("report", help="Pretty-print one recorded tool call.")
    p_report.add_argument("request_id")

    p_verify = sub.add_parser("verify", help="Check audit-log hash-chain integrity.")
    p_verify.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Verify a single day's .jsonl in isolation instead of the full "
        "cross-day trail.",
    )
    p_verify.add_argument(
        "--checkpoint",
        metavar="DATE",
        default=None,
        help="Also verify the Ed25519-signed checkpoint for this date, after "
        "the full trail's hash chain, and print its state. Requires --pubkey.",
    )
    p_verify.add_argument(
        "--pubkey",
        type=Path,
        default=None,
        help="Public key file for --checkpoint verification.",
    )

    p_hold = sub.add_parser(
        "hold", help="Place a legal/retention hold on a calendar day."
    )
    p_hold.add_argument("date", help="YYYY-MM-DD")
    p_hold.add_argument("--reason", default=None)

    p_release_hold = sub.add_parser(
        "release-hold", help="Remove a hold from a calendar day."
    )
    p_release_hold.add_argument("date", help="YYYY-MM-DD")

    p_gc = sub.add_parser(
        "gc",
        help="Delete day files past SQT_AUDIT_RETENTION_DAYS, excluding held "
        "days. Dry-run by default.",
    )
    p_gc.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete candidates. Without this flag, only lists them.",
    )
    p_gc.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="Override SQT_AUDIT_RETENTION_DAYS for this invocation.",
    )

    p_cache = sub.add_parser(
        "cache",
        help="Maintain the OHLCV disk cache. `cache gc` lists (or with "
        "--confirm deletes) files of a dead format generation, which are "
        "never read again, and temp files an interrupted write left behind "
        "over an hour ago; nothing else is evicted.",
    )
    p_cache.add_argument(
        "action",
        choices=["gc"],
        help="gc: dead generations and orphaned temp files.",
    )
    p_cache.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete. Without this flag, only lists the files.",
    )

    p_runs = sub.add_parser(
        "runs",
        help="Maintain the runs directory. `runs gc` lists (or with --confirm "
        "deletes) what interrupted writes left there: temp files, and model or "
        "dataset directories whose registration never committed. Published "
        "values are never collected.",
    )
    p_runs.add_argument(
        "action", choices=["gc"], help="gc: leftovers of interrupted writes."
    )
    p_runs.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete. Without this flag, only lists what would go.",
    )
    p_runs.add_argument(
        "--older-than",
        type=float,
        default=None,
        metavar="HOURS",
        help="Only leftovers untouched for this many hours (default 24), so "
        "a registration or conversion in progress is never collected.",
    )

    p_seal = sub.add_parser(
        "seal", help="Chmod a day file read-only (not WORM — see docs)."
    )
    p_seal.add_argument("date", help="YYYY-MM-DD")

    p_export = sub.add_parser(
        "export",
        help="Package day files in a date range plus the chain index and a "
        "manifest into a zip bundle for an external auditor.",
    )
    p_export.add_argument("--start", required=True, help="YYYY-MM-DD, inclusive")
    p_export.add_argument("--end", required=True, help="YYYY-MM-DD, inclusive")
    p_export.add_argument("--out", type=Path, required=True, help="Output .zip path")

    p_keygen = sub.add_parser(
        "keygen",
        help="Generate an Ed25519 signing keypair (local development only "
        "— not production key custody).",
    )
    p_keygen.add_argument(
        "--out",
        type=Path,
        default=Path("."),
        help="Directory to write the keypair into.",
    )

    p_anchor = sub.add_parser(
        "anchor", help="Sign a checkpoint for a calendar day (Ed25519)."
    )
    p_anchor.add_argument("date", help="YYYY-MM-DD")
    p_anchor.add_argument(
        "--key",
        type=Path,
        default=None,
        help="Private key file (else SQT_AUDIT_SIGNING_KEY_PATH).",
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "replay":
            report, exit_code = _replay(args.request_id, None)
            print(report)
            return exit_code
        elif args.command == "compare":
            print(cmd_compare(args.request_id_a, args.request_id_b))
        elif args.command == "report":
            print(cmd_report(args.request_id))
        elif args.command == "verify":
            if args.checkpoint is not None:
                if args.pubkey is None:
                    print("error: --checkpoint requires --pubkey", file=sys.stderr)
                    return 1
                # The chain is checked as well, every time. On its own the
                # checkpoint check ran INSTEAD of the chain and printed
                # "Signature valid." with exit 0 for a day whose records had
                # been edited; the two catch different things, and a command
                # named `verify` should not be able to pass a trail that
                # `sqt verify` fails. See the CHANGELOG entry of 2026-09-27.
                notes: List[str] = []
                head: Dict[str, Any] = {}
                problems = cmd_verify(notes=notes, head=head)
                print(_format_verify(problems, notes, head))
                found = cmd_verify_checkpoint(args.checkpoint, args.pubkey)
                print(_format_checkpoint(args.checkpoint, found))
                if problems:
                    return 1
                return _CHECKPOINT_EXIT_CODES.get(found.state, 1)
            notes = []
            head = {}
            problems = cmd_verify(file=args.file, notes=notes, head=head)
            print(_format_verify(problems, notes, head))
            return 1 if problems else 0
        elif args.command == "hold":
            path = cmd_hold(args.date, reason=args.reason)
            print(f"Hold placed on {args.date} ({path})")
        elif args.command == "release-hold":
            released = cmd_release_hold(args.date)
            if released:
                print(f"Hold released on {args.date}")
            else:
                print(f"No hold existed on {args.date}")
        elif args.command == "gc":
            dates = cmd_gc(confirm=args.confirm, retention_days=args.retention_days)
            if not dates:
                verb = "deleted" if args.confirm else "eligible for deletion"
                print(f"No day files {verb}.")
            else:
                verb = "Deleted" if args.confirm else "Eligible for deletion (dry-run)"
                print(f"{verb}:")
                for d in dates:
                    print(f"  - {d}")
        elif args.command == "cache":
            dead, orphans = _cache_gc(args.confirm)
            if not dead and not orphans:
                verb = "deleted" if args.confirm else "to collect"
                print(f"No cache files {verb}.")
            if dead:
                verb = "Deleted" if args.confirm else "Dead generation (dry-run)"
                print(f"{verb}: {len(dead)} file(s)")
                for p in dead:
                    print(f"  - {p.name}")
            if orphans:
                verb = (
                    "Deleted orphaned temp files"
                    if args.confirm
                    else "Orphaned temp files (dry-run)"
                )
                print(f"{verb}: {len(orphans)} file(s)")
                for p in orphans:
                    print(f"  - {p.name}")
        elif args.command == "runs":
            report = cmd_runs_gc(confirm=args.confirm, older_than_hours=args.older_than)
            _print_runs_gc(report, args.confirm)
        elif args.command == "seal":
            path = cmd_seal(args.date)
            print(f"Sealed {path} read-only.")
        elif args.command == "export":
            out_path = cmd_export(args.start, args.end, args.out)
            print(f"Exported bundle: {out_path}")
        elif args.command == "keygen":
            priv_path, pub_path = cmd_keygen(args.out)
            print(f"Private key: {priv_path}")
            print(f"Public key:  {pub_path}")
            print(
                "WARNING: local development only — not a production "
                "key-custody solution. See Documentation/10_auditability.md."
            )
        elif args.command == "anchor":
            checkpoint_path = cmd_anchor(args.date, key_path=args.key)
            print(f"Checkpoint signed: {checkpoint_path}")
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
