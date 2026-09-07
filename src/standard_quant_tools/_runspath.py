"""
The runs directory, and the two checks that keep callers inside it.

`backtest.artifacts` and `modeling.artifacts` both build filesystem paths
from identifiers an LLM can choose -- a `run_id` on a compact-backtest
request, a `ds_...` or `mdl_...` artifact id -- and both had their own copy
of the guard against that. Same regex, same runs-directory lookup, same
error message.

TWO COPIES OF A PATH-TRAVERSAL CHECK IS THE WRONG NUMBER, and not because
of the duplication itself. It is because the two copies are only equal
today. Harden one -- a new escape to reject, a case the regex lets through,
a symlink to resolve differently -- and the other keeps the old behaviour
under the same name, which reads as fixed everywhere and is fixed in one
place. A half-applied security fix is worse than an unapplied one: it
removes the reason to look again.

The asymmetry was already there. `backtest.artifacts` carried the docstring
explaining what the check defends against and layered
`_resolved_within_runs_dir` on top of it as defence in depth;
`modeling.artifacts` had the same validator with no explanation, and
open-coded the containment check inside `run_dir` instead of sharing it.

Lives at the top level for the reason `_jsonsafe` does: both need it and
neither should import the other.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from standard_quant_tools.error import ValidationError

#: The environment variable that relocates the runs root.
RUNS_DIR_ENV = "SQT_RUNS_DIR"

#: A plain slug. Deliberately a whitelist rather than a blacklist of
#: dangerous sequences: '..', '/', '\', ':', and a NUL byte are all excluded
#: by not being letters, digits, '_' or '-', and so is whatever the next
#: platform-specific escape turns out to be.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def runs_dir() -> Path:
    """Where artifacts live: `$SQT_RUNS_DIR`, or a cache dir under $HOME."""
    return Path(
        os.environ.get(
            RUNS_DIR_ENV,
            str(Path.home() / ".cache" / "standard_quant_tools" / "runs"),
        )
    )


def validate_identifier(value: str, field_name: str) -> None:
    """
    Refuse anything that is not a plain slug.

    run_id/name are LLM-reachable (e.g. `BacktestCompactInput.run_id`) and
    get joined directly into a filesystem path -- so path separators, '..',
    null bytes, and a drive-letter or absolute prefix are rejected here,
    before they can reach the path at all.
    """
    if not value or not _IDENTIFIER_RE.match(value):
        raise ValidationError(
            f"{field_name}={value!r} is not a valid identifier — only letters, "
            "digits, '_', and '-' are allowed (no path separators, '..', or "
            "empty string)."
        )


def resolve_within_runs_dir(path: Path) -> Path:
    """
    Defence in depth on top of `validate_identifier`: confirm the final
    RESOLVED path is inside the runs root before any read or write.

    The validator works on the identifier and this works on the result, so
    a way of building a path that never passes through a validated
    identifier -- a symlink inside the runs directory, a caller assembling
    a path itself -- still has to land inside the root.
    """
    root = runs_dir().resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValidationError(
            f"resolved path {resolved} escapes {RUNS_DIR_ENV} ({root})"
        )
    return resolved
