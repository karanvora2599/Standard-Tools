"""
What this session actually wrote to disk.

The artifact store is a complete storage layer -- atomic write, content
hash, key validation, root containment, listing -- and the tool surface
touched it zero times, including on the calls that had just written a file
through it. A run that produced an equity curve, a trades table and a
model package handed back three URIs in three different responses, and
once those scrolled out of the conversation the files were unreachable:
nothing could say what a run id held, and `describe_artifact` needed a URI
it no longer had.

So this is the listing. A run id narrows it to one run, which is the usual
question -- what did that run leave behind -- and the default is
everything the store holds, newest run first by key order.

THE HASH IS OPT-IN because it is the only part that reads the files. A
listing is a directory walk and a stat per key; hashing opens and streams
every one of them, and on a store of any size that is a different order of
cost for an answer most callers do not need. When it is asked for, it is
the store's own digest -- the same 16 hexadecimal characters
`describe_artifact` reports -- so the two can be compared directly rather
than being two hashes of one file that never match.

Reads only. Nothing here writes, moves or removes an artifact.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class _Result(BaseModel):
    model_config = ConfigDict(extra="allow")

    warnings: List[str] = Field(default_factory=list)


class ListArtifactsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Optional[str] = Field(
        None,
        description=(
            "Narrow to one run. A run id is the slug a tool was given when "
            "it published -- letters, digits, underscore and hyphen. Omit "
            "for every artifact the store holds."
        ),
    )
    include_hash: bool = Field(
        False,
        description=(
            "Also report each artifact's content hash. Off by default "
            "because it is the only part that opens the files: a listing is "
            "a stat per key, hashing is a full read of every one. Ask for "
            "it when you need to confirm two runs produced the same bytes."
        ),
    )
    limit: int = Field(
        200,
        ge=1,
        le=5000,
        description=(
            "Cap on returned entries. `total` and `truncated` say whether "
            "narrowing by run id would show you the rest."
        ),
    )


class ArtifactEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    key: str = Field(
        "",
        description=(
            "The store key, '<run_id>/<filename>'. describe_artifact takes "
            "this directly, as well as the absolute uri beside it."
        ),
    )
    uri: str = ""
    size_bytes: int = 0
    modified_utc: Optional[str] = None
    content_hash: Optional[str] = Field(
        None,
        description=(
            "The store's digest -- SHA-256 truncated to 16 hexadecimal "
            "characters, the same value describe_artifact reports, so the "
            "two compare directly. Null unless include_hash was set."
        ),
    )


class ListArtifactsResult(_Result):
    runs_dir: str = ""
    n_artifacts: int = 0
    total: int = 0
    truncated: bool = False
    n_runs: int = 0
    artifacts: List[ArtifactEntry] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


def list_artifacts(input_data: ListArtifactsInput) -> ListArtifactsResult:
    """
    Every artifact this library has persisted, or just one run's.

    Tools that write a file hand back a URI once, in one response. After
    that the file existed and nothing could find it: there was no way to
    ask what a run id held, and describing an artifact required a URI that
    had already scrolled away. This lists them.

    The content hash is opt-in because it is the only part that reads the
    files rather than their directory entries. Asked for, it is the store's
    own 16-character digest -- the same one `describe_artifact` reports, so
    a listing and a description of the same file agree.

    Read-only: nothing is written, moved or removed. An unusable run id is
    refused by name rather than being silently treated as 'everything'.
    """
    from standard_quant_tools.artifact_store import LocalArtifactStore

    store = LocalArtifactStore()
    root = store.root
    # A bad run id is refused HERE, by the store's own key rule, rather
    # than quietly listing the whole store: a caller who mistyped a run id
    # and got every artifact back would read it as "that run produced all
    # of this".
    keys = store.list(input_data.run_id or "")

    entries: List[ArtifactEntry] = []
    warnings: List[str] = []
    # Counted over EVERY key, not over the listed page, so a truncated
    # listing still says how many runs the store holds.
    runs = {key.split("/", 1)[0] for key in keys}
    for key in keys[: input_data.limit]:
        uri = store.uri(key)
        path = Path(uri)
        try:
            stat = path.stat()
            size = int(stat.st_size)
            modified = (
                datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except OSError:
            # The file went away between the listing and the stat. Report
            # the key rather than dropping it: a key with no file is a
            # finding, and hiding it would make the listing look clean.
            size, modified = 0, None
            warnings.append(
                f"{key} was listed but could not be read. It was removed "
                "between the listing and this call, or the process lost "
                "access to it."
            )
        digest: Optional[str] = None
        if input_data.include_hash and modified is not None:
            try:
                digest = store.hash(key)
            except Exception as exc:  # noqa: BLE001 - one key, not the call
                logger.debug("[list_artifacts] hash failed for %s", key)
                warnings.append(f"{key} could not be hashed: {exc}")
        entries.append(
            ArtifactEntry(
                key=key,
                uri=uri,
                size_bytes=size,
                modified_utc=modified,
                content_hash=digest,
            )
        )

    truncated = len(keys) > input_data.limit
    notes: List[str] = [
        "`key` is what describe_artifact takes, as well as the absolute "
        "uri beside it -- either resolves to the same file inside the "
        "store root.",
    ]
    if not input_data.include_hash and entries:
        notes.append(
            "Content hashes were not computed. Set include_hash to confirm "
            "that two artifacts hold the same bytes; it reads every file, "
            "which is why it is a flag."
        )
    if truncated:
        notes.append(
            f"{len(keys)} artifacts exist and the first {input_data.limit} "
            "are listed. Narrow by run id rather than raising the limit."
        )

    if not keys:
        if input_data.run_id:
            warnings.append(
                f"run {input_data.run_id!r} holds no artifacts under "
                f"{root}. Either nothing was published under that id, or "
                "this process is pointed at a different store than the one "
                "that wrote them -- describe_effective_config reports which."
            )
        else:
            warnings.append(
                f"No artifacts under {root}. A tool writes one only when it "
                "is given a run id, and a process with a different "
                "SQT_RUNS_DIR cannot see another's -- "
                "describe_effective_config reports the root in force."
            )

    logger.debug(
        "[list_artifacts] root=%s keys=%d listed=%d",
        root,
        len(keys),
        len(entries),
    )
    return ListArtifactsResult(
        runs_dir=str(root),
        n_artifacts=len(entries),
        total=len(keys),
        truncated=truncated,
        n_runs=len(runs),
        artifacts=entries,
        notes=notes,
        warnings=warnings,
    )


__all__ = [
    "ArtifactEntry",
    "ListArtifactsInput",
    "ListArtifactsResult",
    "list_artifacts",
]
