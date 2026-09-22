"""export_bundle(): package a date range of day files, the chain index, the
signed-checkpoint sidecars of those days, a manifest, and the standalone
dependency-free verifier into one zip -- the artifact meant to be handed to
an external auditor."""

import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .paths import _INDEX_FILENAME, _audit_dir, _iter_day_files
from .provenance import _git_sha, _package_version

_EXPORT_README = """\
Standard Quant Tools -- exported audit trail bundle
====================================================

Contents:
  - one YYYY-MM-DD.jsonl file per day in the exported date range
  - _chain_index.jsonl (if present): the cross-day hash-chain witness log
  - YYYY-MM-DD.checkpoint.json / .checkpoint.sig, for each exported day
    that was anchored with an Ed25519 signed checkpoint
  - manifest.json: per-file SHA-256 hashes, record counts, and provenance
    (package version, git commit, generation timestamp)
  - verify_audit_log.py: a standalone, dependency-free verifier (Python
    standard library only -- no need to install this project or pandas/
    numpy/pydantic to run it)

To verify this bundle:
  1. Confirm each file's SHA-256 in manifest.json matches the file on disk
     (e.g. `sha256sum *.jsonl` on Linux/macOS, `certutil -hashfile <file>
     SHA256` on Windows) -- this confirms the bundle itself wasn't altered
     since export.
  2. Run: python verify_audit_log.py .
     This independently re-walks every record's hash chain and the
     cross-day chain index and reports any tamper-evidence problems.

What a clean result does and does not prove: it confirms the exported
records are internally self-consistent and match their hash chain. It
does NOT prove the source system's filesystem was never tampered with
before export.

Ed25519 checkpoint signing is implemented, and a day that was anchored
with it travels with its .checkpoint.json and .checkpoint.sig sidecars,
listed in manifest.json like every other file here. A checkpoint is an
anchor OUTSIDE the JSONL files: it fixes a day's final record hash and
its chain-index entry at signing time, so a wholesale rewrite of that day
-- the one thing the hash chain alone cannot rule out -- fails against
it. The public key that checks those signatures deliberately does NOT
travel in this bundle; obtain it out of band from whoever runs the
signing key, since a signature and the key that verifies it shipped in
the same envelope prove nothing about who produced either. Days with no
sidecars here were never anchored.

Treat this as engineering evidence supporting an audit, not a legal
attestation by itself.
"""


def _checkpoint_sidecars(directory: Path, date: str) -> List[Path]:
    """The signed-checkpoint files for one day, in the order they are
    written: the checkpoint payload and its detached signature. Empty for a
    day that was never anchored, which is the ordinary case — signing is
    optional, and a missing sidecar is not a problem to report here."""
    return [
        p
        for p in (
            directory / f"{date}.checkpoint.json",
            directory / f"{date}.checkpoint.sig",
        )
        if p.exists()
    ]


@dataclass(frozen=True)
class ExportedBundle:
    """
    What `export_bundle` wrote: the zip's path, and how much of the audit
    log went into it.

    A bundle over a date range that matched nothing is still a well-formed
    zip of a manifest, a README and a verifier — the same size, the same
    shape, and empty. `day_files` and `record_count` are what tell those
    two apart, so a caller can refuse an export of nothing instead of
    handing an auditor a bundle with no audit in it.

    Passes as a path (`os.fspath`, `Path(...)`, `str(...)`) so callers that
    only ever wanted where it landed keep working.
    """

    path: Path
    day_files: int
    record_count: int

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)


def export_bundle(
    start_date: str,
    end_date: str,
    out_path: Union[str, Path],
    audit_dir: Optional[Union[str, Path]] = None,
) -> ExportedBundle:
    """
    Package every day file in `[start_date, end_date]` (inclusive,
    "YYYY-MM-DD") plus the chain index, any signed-checkpoint sidecars for
    the exported days, a manifest, a copy of the standalone verifier
    script, and verification instructions into one zip — the artifact meant
    to be handed to an external auditor.

    The `<date>.checkpoint.json`/`.checkpoint.sig` sidecars travel when
    they exist (see the CHANGELOG entry of 2026-09-22): they are the only
    evidence in the directory that survives a wholesale rewrite of a day
    file, so a bundle that left them behind handed the auditor strictly
    less than the source system had. The public key is not included and is
    not this library's to distribute.

    Returns an `ExportedBundle`: `out_path`, plus the number of day files
    and records it holds.
    """
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    day_files = [
        p for p in _iter_day_files(directory) if start_date <= p.stem <= end_date
    ]
    index_path = directory / _INDEX_FILENAME
    # parents[0]=audit, [1]=standard_quant_tools, [2]=src, [3]=repo root.
    verifier_script = (
        Path(__file__).resolve().parents[3] / "scripts" / "verify_audit_log.py"
    )

    manifest: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "package_version": _package_version(),
        "git_commit_sha": _git_sha(),
        "day_files": len(day_files),
        "record_count": 0,
        "files": [],
    }

    def _describe(p: Path) -> Dict[str, Any]:
        content = p.read_bytes()
        # Only the JSONL files hold records; a checkpoint sidecar's line
        # count is not a record count and saying so would be a small lie
        # in the one file whose job is to be exact.
        record_count = (
            sum(1 for line in content.decode("utf-8").splitlines() if line.strip())
            if p.suffix == ".jsonl"
            else None
        )
        return {
            "name": p.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "record_count": record_count,
        }

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in day_files:
            described = _describe(p)
            manifest["record_count"] += described["record_count"] or 0
            manifest["files"].append(described)
            zf.write(p, arcname=p.name)
            for sidecar in _checkpoint_sidecars(directory, p.stem):
                manifest["files"].append(_describe(sidecar))
                zf.write(sidecar, arcname=sidecar.name)
        if index_path.exists():
            manifest["files"].append(_describe(index_path))
            zf.write(index_path, arcname=index_path.name)
        if verifier_script.exists():
            zf.write(verifier_script, arcname="verify_audit_log.py")
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        zf.writestr("README.txt", _EXPORT_README)

    return ExportedBundle(
        path=out_path,
        day_files=manifest["day_files"],
        record_count=manifest["record_count"],
    )
