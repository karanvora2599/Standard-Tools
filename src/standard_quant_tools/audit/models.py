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
    # Where `duration_ms` went. `fetch_ms` is the sum of the `fetch_ms` each
    # data source carries -- the call's time from its start to its last
    # completed data access -- and `compute_ms` is `duration_ms - fetch_ms`,
    # floored at 0. A FAILED call has no `compute_ms`: a fetch that fails
    # never reports itself, so the time after the last completed access
    # may be a vendor that never answered rather than computation. Both
    # None for records written before the split existed; those still
    # verify, because a record is hashed as it was stored.
    fetch_ms: Optional[float] = None
    compute_ms: Optional[float] = None
    output_hash: Optional[str] = None
    # The same output hashed with run-specific dataset/model identifiers
    # normalized away (see replay.normalize_identifiers). Modeling mints a
    # fresh id per run and embeds it in artifact paths, so the literal
    # output_hash above can never reproduce for those tools; this is what
    # replay actually compares for them. None for records written before
    # this field existed, which replay reports as "not comparable" rather
    # than as a mismatch.
    output_hash_normalized: Optional[str] = None
    # The output hashed once more with run-specific identifiers normalized
    # away (as above) AND every float rounded to twelve significant digits
    # (hashing.round_floats). The exact hash is bit-for-bit, and bits are
    # only promised on the same native build and instruction-set path: the
    # AVX2+FMA and scalar reductions round differently. This is the hash
    # replay falls back to when the exact one misses on a different build
    # or path, so it can say "reproduced to twelve digits" rather than
    # "the code changed". None for records written before it existed.
    output_hash_rounded: Optional[str] = None
    status: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    # Reproducibility provenance — None when unavailable (e.g. no git
    # checkout), never a reason to fail the call itself.
    git_commit_sha: Optional[str] = None
    package_version: Optional[str] = None
    random_seed: Optional[int] = None
    strategy_source_hash: Optional[str] = None
    # WHICH compiled build ran, not just whether one did: the import-time
    # verdict and the short source digest the extension was stamped with,
    # e.g. "match:df27c6e4af54", "stale:0a1b2c3d4e5f" (present, refused, the
    # Python path ran), "absent" or "disabled". `cpp_available` alone was
    # true for an extension weeks older than the code calling it. None for
    # records written before this field existed; those still verify,
    # because a record is hashed as it was stored.
    native_build: Optional[str] = None
    # WHICH instruction-set path the compiled kernels took on the machine
    # that wrote the record: "avx2+fma" or "scalar", or "none" when no
    # extension ran. A property of the CPU, not of the build -- one binary
    # takes either path -- and the paths agree to twelve significant
    # digits, not bit for bit. None for records written before the field.
    native_isa: Optional[str] = None
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
    # The hashes the comparison was actually made on. `output_match` is the
    # verdict; these are the evidence behind it, and without them "the
    # output changed" is a claim the caller cannot check, narrow down or
    # quote. `new_output_hash` is what the replay produced,
    # `stored_output_hash` what the record carried, and
    # `new_output_hash_normalized` the replay's hash with run-specific
    # dataset/model identifiers normalized away — set only when that
    # second comparison was the one that decided the verdict (see
    # replay.normalize_identifiers). Each is None when the arm that runs
    # did not compute it: a replay of a call that FAILED originally has no
    # stored output to compare against. Per-data-source `old_hash`/
    # `new_hash` live on each entry of `data_source_matches`.
    new_output_hash: Optional[str] = None
    stored_output_hash: Optional[str] = None
    new_output_hash_normalized: Optional[str] = None
    # The twelve-significant-digit comparison, made only when the exact one
    # missed. `rounded_output_match` is None when it was not made: the exact
    # hash matched, or the record predates the rounded hash.
    rounded_output_match: Optional[bool] = None
    new_output_hash_rounded: Optional[str] = None
    stored_output_hash_rounded: Optional[str] = None
    # How the build that wrote the record differs from the one replaying it
    # ("native_build: recorded 'match:…', now 'match:…'", or "not recorded"
    # for a record that predates the field). Filled whenever the exact hash
    # missed; empty means same build on the same instruction-set path.
    build_differences: List[str] = field(default_factory=list)
