"""
Decision-record audit trail for agent-tool calls.

Every call routed through `agent.tools.dispatch()` can produce an immutable
JSONL record capturing its inputs, the market data it pulled (with content
hashes), which execution path ran, and a hash of its output. `verify_replay()`
re-runs a recorded call and reports whether the data and output still match —
enough to tell a stale/tampered cache apart from a genuine code change.

Nothing here runs automatically. The package only ever attaches a
`NullHandler` by default (see `standard_quant_tools/__init__.py`); call
`configure_logging()` explicitly to see log output, and set
`SQT_AUDIT_ENABLED=0` to disable decision-record writes entirely.

This is a package, not a single module, split by concern so it stays
readable as features accrete (retention/legal-hold, checkpoint signing,
pluggable storage backends, ...):

    hashing     — content-fingerprint hashing (hash_payload, hash_dataframe)
    context     — per-call request context + correlated-logging helper
    provenance  — git/package-version/strategy-source best-effort provenance
    paths       — audit-dir resolution, day-file discovery, advisory locking
    models      — DecisionRecord, ReplayResult
    storage     — AuditStorageBackend, LocalFilesystemBackend (pluggable I/O)
    writer      — AuditWriter (hash-chained, fsync'd JSONL + chain index)
    verify      — verify_audit_log_integrity, verify_audit_trail_integrity
    redaction   — SQT_AUDIT_REDACT_FIELDS field redaction
    retention   — legal hold, retention/gc, read-only sealing
    export      — export_bundle (auditor-ready zip)
    signing     — Ed25519 checkpoint signing (optional `cryptography` extra)
    dispatch    — _run_and_record, the core agent.tools.dispatch() uses
    replay      — verify_replay

Everything below is re-exported here so `standard_quant_tools.audit.<name>`
keeps working exactly as it did when this was a single module — this
`__init__.py` is the only place that needs to know how the package is
internally organized.
"""

from .context import (
    RequestIdFilter,
    configure_logging,
    new_request_id,
    record_data_access,
)
from .dispatch import _run_and_record
from .export import export_bundle
from .hashing import hash_dataframe, hash_payload
from .models import DecisionRecord, ReplayResult
from .paths import (
    _DAY_FILE_RE,
    _GENESIS_HASH,
    _INDEX_FILENAME,
    _audit_dir,
    _audit_enabled,
    _iter_day_files,
)
from .provenance import (
    _cpp_available,
    _git_sha,
    _package_version,
    _strategy_source_hash,
)
from .redaction import _redact, _redact_fields, redact_text
from .replay import verify_replay
from .retention import gc, gc_candidates, hold_day, is_held, release_hold, seal_day
from .signing import (
    HAS_CRYPTOGRAPHY,
    checkpoint_and_sign,
    generate_keypair,
    verify_checkpoint_signature,
)
from .storage import AuditStorageBackend, LocalFilesystemBackend
from .verify import verify_audit_log_integrity, verify_audit_trail_integrity
from .writer import AuditWriter

__all__ = [
    "AuditStorageBackend",
    "AuditWriter",
    "DecisionRecord",
    "HAS_CRYPTOGRAPHY",
    "LocalFilesystemBackend",
    "ReplayResult",
    "RequestIdFilter",
    "checkpoint_and_sign",
    "configure_logging",
    "export_bundle",
    "gc",
    "gc_candidates",
    "generate_keypair",
    "hash_dataframe",
    "hash_payload",
    "hold_day",
    "is_held",
    "new_request_id",
    "record_data_access",
    "release_hold",
    "seal_day",
    "verify_audit_log_integrity",
    "verify_audit_trail_integrity",
    "verify_checkpoint_signature",
    "verify_replay",
]
