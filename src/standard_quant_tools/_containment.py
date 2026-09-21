"""
One containment check for every root this library writes under.

Four call sites -- the OHLCV disk cache, the artifact store, the runs
directory and the audit-bundle export -- each confirmed that a resolved
path lies inside its root, and one of the four knew about Windows'
extended-length prefix. `Path.resolve()` returns the `\\\\?\\` form for a
path that exists on disk and the plain form for one that does not, so a
cold runs directory compared a prefixed root against an unprefixed child
and refused it as a traversal (findings, the plumbing). The prefix is
handled here, once, and the four sites call this.
"""

from __future__ import annotations

from pathlib import Path

from standard_quant_tools.error import ValidationError

_EXTENDED_LENGTH_PREFIX = "\\\\?\\"


def _comparable(path: Path) -> Path:
    """The path without Windows' extended-length prefix, for comparison
    only: the prefixed form is what the filesystem APIs are handed."""
    return Path(str(path).removeprefix(_EXTENDED_LENGTH_PREFIX))


def is_within(resolved: Path, root: Path) -> bool:
    """Whether `resolved` lies inside `root`, prefix or no prefix."""
    return _comparable(Path(resolved)).is_relative_to(_comparable(Path(root)))


def require_within(resolved: Path, root: Path, message: str) -> Path:
    """`resolved` itself when it lies inside `root`; a ValidationError
    carrying `message` otherwise."""
    if not is_within(resolved, root):
        raise ValidationError(message)
    return resolved


__all__ = ["is_within", "require_within"]
