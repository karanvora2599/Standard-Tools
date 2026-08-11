"""Data shapes: the `DecisionRecord` written to a day's JSONL file per tool
call, and the `ReplayResult` returned by `verify_replay()`."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DecisionRecord(BaseModel):
    request_id: str
    timestamp_utc: str
    tool_name: str
    input: Dict[str, Any]
    data_sources: List[Dict[str, Any]] = Field(default_factory=list)
    cpp_available: bool
    n_workers: Optional[int] = None
    duration_ms: float
    output_hash: Optional[str] = None
    # The same output hashed with run-specific dataset/model identifiers
    # normalized away (see replay.normalize_identifiers). Modeling mints a
    # fresh id per run and embeds it in artifact paths, so the literal
    # output_hash above can never reproduce for those tools; this is what
    # replay actually compares for them. None for records written before
    # this field existed, which replay reports as "not comparable" rather
    # than as a mismatch.
    output_hash_normalized: Optional[str] = None
    status: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    # Reproducibility provenance — None when unavailable (e.g. no git
    # checkout), never a reason to fail the call itself.
    git_commit_sha: Optional[str] = None
    package_version: Optional[str] = None
    random_seed: Optional[int] = None
    strategy_source_hash: Optional[str] = None
    # Hash-chain tamper-evidence: each record's hash covers its own content
    # plus the previous record's hash, so editing a past line changes that
    # line's hash and breaks the chain for every record after it (unless an
    # attacker also rewrites every subsequent line to match — this detects
    # accidental/partial tampering, not a fully-rewritten log; there is no
    # external anchor/signature to detect a wholesale rewrite). "0" * 16 for
    # the first record of a day's file.
    prev_record_hash: Optional[str] = None
    record_hash: Optional[str] = None


@dataclass
class ReplayResult:
    request_id: str
    tool_name: str
    output_match: Optional[bool]
    data_source_matches: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
