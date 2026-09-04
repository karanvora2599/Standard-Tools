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
                                            stdout, one per line).
    sqt hold <date> [--reason TEXT]      — place a legal/retention hold on
                                            a calendar day (YYYY-MM-DD),
                                            protecting it from `sqt gc`.
    sqt release-hold <date>              — remove a hold from a day.
    sqt gc [--confirm]                   — delete day files past
                                            SQT_AUDIT_RETENTION_DAYS,
                                            excluding held days. Dry-run
                                            (lists candidates only) unless
                                            --confirm is passed.
    sqt seal <date>                      — chmod a day file read-only
                                            (not WORM — see
                                            audit.seal_day's docstring).
    sqt export --start D --end D --out F — package day files in [start,
                                            end] plus the chain index, a
                                            manifest, and the standalone
                                            verifier into one zip bundle.
    sqt keygen [--out DIR]                — generate an Ed25519 signing
                                            keypair. Local development only
                                            — not production key custody.
    sqt anchor <date> [--key PATH]       — sign a checkpoint for a calendar
                                            day (see audit.checkpoint_and_sign).
                                            Key from --key or
                                            SQT_AUDIT_SIGNING_KEY_PATH.
    sqt verify --checkpoint <date>
               --pubkey PATH             — verify a checkpoint's Ed25519
                                            signature using only the public
                                            key. Exit 0 if valid, 1 otherwise.

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
    decision records."""
    directory = audit_dir if audit_dir is not None else audit._audit_dir()
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.jsonl")):
        if not audit._DAY_FILE_RE.match(path.name):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)


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
    file: Optional[Path] = None, audit_dir: Optional[Path] = None
) -> List[str]:
    """
    Check hash-chain integrity. `file` (a single day's .jsonl) checks just
    that file in isolation; no `file` checks the full cross-day trail
    (every day file plus the chain index) rooted at `audit_dir`.

    Returns a list of human-readable problems (empty if clean).
    """
    if file is not None:
        return audit.verify_audit_log_integrity(file)
    return audit.verify_audit_trail_integrity(audit_dir)


def _format_verify(problems: List[str]) -> str:
    if not problems:
        return "OK — no integrity problems found."
    lines = [f"{len(problems)} problem(s) found:"]
    lines.extend(f"  - {p}" for p in problems)
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


def cmd_seal(date: str, audit_dir: Optional[Path] = None) -> Path:
    return audit.seal_day(date, audit_dir=audit_dir)


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
) -> bool:
    return audit.verify_checkpoint_signature(date, pubkey, audit_dir=audit_dir)


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
        help="Verify an Ed25519-signed checkpoint for this date instead of "
        "the hash chain. Requires --pubkey.",
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
                ok = cmd_verify_checkpoint(args.checkpoint, args.pubkey)
                print(
                    "Signature valid."
                    if ok
                    else "Signature invalid, or checkpoint/signature file not found."
                )
                return 0 if ok else 1
            problems = cmd_verify(file=args.file)
            print(_format_verify(problems))
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
