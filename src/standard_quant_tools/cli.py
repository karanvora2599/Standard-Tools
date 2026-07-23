"""
Command-line interface for the audit trail's JSONL decision records
(audit.py). Three subcommands, each operating on records by request_id:

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

stdlib argparse only — no new dependency, matching this repo's minimal-
dependency stance. Each subcommand's logic lives in its own `cmd_*` function
so it can be tested directly without spawning a subprocess.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from standard_quant_tools import audit


def _iter_records(audit_dir: Optional[Path] = None) -> Iterator[Dict[str, Any]]:
    directory = audit_dir if audit_dir is not None else audit._audit_dir()
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.jsonl")):
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


def cmd_replay(request_id: str, audit_dir: Optional[Path] = None) -> str:
    """Re-run the recorded call via audit.verify_replay() and format the result."""
    record = find_record(request_id, audit_dir)
    result = audit.verify_replay(record)
    return _format_replay(result)


def cmd_compare(request_id_a: str, request_id_b: str, audit_dir: Optional[Path] = None) -> str:
    """Human-readable diff of two records' status/output/provenance/inputs."""
    a = find_record(request_id_a, audit_dir)
    b = find_record(request_id_b, audit_dir)

    lines = [f"Comparing {request_id_a} vs {request_id_b}", ""]
    fields = [
        "tool_name", "status", "output_hash", "duration_ms",
        "git_commit_sha", "package_version", "strategy_source_hash", "random_seed",
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sqt", description="standard_quant_tools audit-trail CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_replay = sub.add_parser("replay", help="Re-run a recorded tool call and check data/output match.")
    p_replay.add_argument("request_id")

    p_compare = sub.add_parser("compare", help="Diff two recorded tool calls.")
    p_compare.add_argument("request_id_a")
    p_compare.add_argument("request_id_b")

    p_report = sub.add_parser("report", help="Pretty-print one recorded tool call.")
    p_report.add_argument("request_id")

    args = parser.parse_args(argv)

    try:
        if args.command == "replay":
            record = find_record(args.request_id, None)
            result = audit.verify_replay(record)
            print(_format_replay(result))
            return _replay_exit_code(result)
        elif args.command == "compare":
            print(cmd_compare(args.request_id_a, args.request_id_b))
        elif args.command == "report":
            print(cmd_report(args.request_id))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
