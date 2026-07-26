"""export_bundle(): package a date range of day files, the chain index, a
manifest, and the standalone dependency-free verifier into one zip -- the
artifact meant to be handed to an external auditor."""

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .paths import _INDEX_FILENAME, _audit_dir, _iter_day_files
from .provenance import _git_sha, _package_version

_EXPORT_README = """\
Standard Quant Tools -- exported audit trail bundle
====================================================

Contents:
  - one YYYY-MM-DD.jsonl file per day in the exported date range
  - _chain_index.jsonl (if present): the cross-day hash-chain witness log
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
before export, and this bundle carries no cryptographic signature (a
signed-checkpoint feature is planned but not yet implemented). Treat this
as engineering evidence supporting an audit, not a legal attestation by
itself.
"""


def export_bundle(
    start_date: str,
    end_date: str,
    out_path: Union[str, Path],
    audit_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Package every day file in `[start_date, end_date]` (inclusive,
    "YYYY-MM-DD") plus the chain index, a manifest, a copy of the
    standalone verifier script, and verification instructions into one zip
    — the artifact meant to be handed to an external auditor.

    Returns `out_path`.
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
        "files": [],
    }

    def _describe(p: Path) -> Dict[str, Any]:
        content = p.read_bytes()
        record_count = sum(
            1 for line in content.decode("utf-8").splitlines() if line.strip()
        )
        return {
            "name": p.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "record_count": record_count,
        }

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in day_files:
            manifest["files"].append(_describe(p))
            zf.write(p, arcname=p.name)
        if index_path.exists():
            manifest["files"].append(_describe(index_path))
            zf.write(index_path, arcname=index_path.name)
        if verifier_script.exists():
            zf.write(verifier_script, arcname="verify_audit_log.py")
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        zf.writestr("README.txt", _EXPORT_README)

    return out_path
