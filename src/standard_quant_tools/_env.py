"""
Reading this library's environment settings, one way.

Three readers disagreed about what an EMPTY variable means. The audit
directory treated `VAR=` as unset and used its default; the runs directory
and the OHLCV cache treated it as the path `""`, which is the process
working directory, so one stray `=` in a launcher moved the artifact store
somewhere nobody chose; and the audit switch treated it as "off", which
silently stopped the decision log. Booleans had three vocabularies beside
that, two of them stripping whitespace and one not, and none of them
refused a typo: `flase` read as whatever the reader's fallback happened to
be.

These helpers are the one reading. The rules, and why:

  BLANK MEANS UNSET. Empty or whitespace-only is the default, never a value.
  Nobody sets a variable to three spaces on purpose.

  A RELATIVE PATH IS REFUSED, not anchored. The only anchor available is the
  working directory, which the process that launched this one chose (an MCP
  client picks it) and which `os.chdir` moves mid-run -- so a relative root
  names a different directory depending on when it is read. `~` is
  expanded, because it names one directory for the life of the process.

  AN UNKNOWN WORD IS REFUSED. A flag reads `1/true/yes/on` or
  `0/false/no/off`, any case, padded or not. Anything else raises, so a
  misspelling surfaces in the report that describes the configuration
  instead of quietly meaning "off".

  A REFUSAL NAMES THE VARIABLE AND NEVER ITS VALUE. Some of these settings
  are key paths and credentials, and a refusal is written verbatim into the
  decision log, which cannot be edited afterwards. The operator who set the
  value can read it in their own environment; the message says what is
  wrong with it.

  A LOCAL `.env` IS LOADED FIRST. Every helper calls `config.load_env()`
  before reading, so a value supplied there is in force from the first
  setting read rather than from whichever reader happened to load it.
  `load=False` skips that for a reader that runs at import time, where
  loading a file is a side effect nobody asked for.

Nothing here creates a directory or touches a file beyond asking whether an
existing path is a file or a directory.

Top-level for the reason `_runspath` and `_containment` are: the audit
package, the data layer, the model registry and the server all read settings,
and none of them should import another to do it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

#: The words a flag may be set to, compared after stripping and lowercasing.
TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
FALSE_WORDS = frozenset({"0", "false", "no", "off"})

_PATH_KINDS = ("dir", "file")

PathDefault = Union[Path, Callable[[], Path], None]


def _load_env() -> None:
    """Load a local `.env` once per process; a failure is not a setting."""
    try:
        from standard_quant_tools.config import load_env

        load_env()
    except Exception:  # noqa: BLE001 - a missing or unreadable .env is normal
        logger.debug("[_env] load_env failed", exc_info=True)


def env_str(name: str, *, load: bool = True) -> Optional[str]:
    """
    The variable's value with surrounding whitespace removed, or None when
    it is unset, empty or whitespace-only.
    """
    if load:
        _load_env()
    raw = os.environ.get(name)
    if raw is None:
        return None
    text = raw.strip()
    return text or None


def _default_path(default: PathDefault) -> Optional[Path]:
    if default is None:
        return None
    return default() if callable(default) else Path(default)


def _checked_path(name: str, text: str, *, kind: str, where: str = "") -> Path:
    """One path-valued entry, by the rules in the module docstring.

    `where` places the entry inside a list ("entry 2 of 3 in "), so a
    refusal about a list says which entry without echoing any of them.
    """
    if kind not in _PATH_KINDS:
        raise ValueError(f"kind must be one of {_PATH_KINDS}, got {kind!r}")
    if "\x00" in text:
        raise ValidationError(
            f"{where}{name} contains a NUL character, which no path can hold. "
            "Correct it in the environment this process starts with."
        )
    try:
        path = Path(text).expanduser()
    except RuntimeError as exc:  # no home directory to expand `~` against
        raise ValidationError(
            f"{where}{name} starts with `~`, and this process has no home "
            "directory to expand it against. Give the absolute path."
        ) from exc
    if not path.is_absolute():
        raise ValidationError(
            f"{where}{name} is a relative path. A relative path would be read "
            "against the working directory, which the program that launched "
            "this one chose and which changes whenever the process changes "
            "directory, so it names a different place depending on when it "
            f"is read. Set {name} to an absolute path, or leave it unset for "
            "the default."
        )
    try:
        if kind == "dir" and path.exists() and not path.is_dir():
            raise ValidationError(
                f"{where}{name} names an existing file, not a directory. It "
                "has to be a directory, or a path where one may be created."
            )
        if kind == "file" and path.is_dir():
            raise ValidationError(
                f"{where}{name} names a directory, not a file. It has to "
                "name the file itself."
            )
    except OSError as exc:
        raise ValidationError(
            f"{where}{name} could not be examined ({type(exc).__name__}). "
            "Check that it names a readable location."
        ) from exc
    return path


def env_path(
    name: str,
    default: PathDefault = None,
    *,
    kind: str = "dir",
    load: bool = True,
) -> Optional[Path]:
    """
    An absolute path from `name`, with `~` expanded.

    Unset, empty or whitespace-only gives `default` (a path, or a callable
    returning one; None when there is no default). A relative path is
    refused by name. `kind="dir"` also refuses a value naming an existing
    file, and `kind="file"` one naming an existing directory. A path that
    does not exist yet is accepted: whether to create it is the caller's
    decision, not this reader's.
    """
    text = env_str(name, load=load)
    if text is None:
        return _default_path(default)
    return _checked_path(name, text, kind=kind)


def env_paths(name: str, *, kind: str = "dir", load: bool = True) -> Tuple[Path, ...]:
    """
    Several absolute paths from one variable, separated by `os.pathsep`
    (`;` on Windows, `:` elsewhere -- the convention PATH uses).

    Unset, empty or whitespace-only gives an empty tuple, and so do empty
    entries (a trailing separator is not a directory). Each entry follows
    `env_path`'s rules, and a refusal says which entry by position. The same
    directory listed twice is kept once.
    """
    text = env_str(name, load=load)
    if text is None:
        return ()
    entries = [entry.strip() for entry in text.split(os.pathsep)]
    entries = [entry for entry in entries if entry]
    paths = []
    for position, entry in enumerate(entries, start=1):
        where = f"entry {position} of {len(entries)} in " if len(entries) > 1 else ""
        path = _checked_path(name, entry, kind=kind, where=where)
        if path not in paths:
            paths.append(path)
    return tuple(paths)


def env_flag(name: str, default: bool, *, load: bool = True) -> bool:
    """
    A boolean from `name`: `default` when unset or blank, otherwise one of
    `TRUE_WORDS` or `FALSE_WORDS` in any case. Any other word is refused
    rather than read as either answer.
    """
    text = env_str(name, load=load)
    if text is None:
        return bool(default)
    word = text.lower()
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    raise ValidationError(
        f"{name} is set to a word that is neither on nor off. It accepts "
        f"{sorted(TRUE_WORDS)} for on and {sorted(FALSE_WORDS)} for off, in "
        "any case; leave it unset for the default. A misspelling is refused "
        "rather than read as either, because a setting that silently means "
        "the opposite of what was typed is found only by its consequences."
    )


def env_int(
    name: str,
    default: Optional[int] = None,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
    load: bool = True,
) -> Optional[int]:
    """
    An integer from `name`: `default` when unset or blank. A value that is
    not an integer, or lies outside `[minimum, maximum]`, is refused by name.
    """
    text = env_str(name, load=load)
    if text is None:
        return default
    try:
        value = int(text)
    except ValueError:
        raise ValidationError(
            f"{name} is not an integer. Set it to a whole number, or leave it "
            "unset for the default."
        ) from None
    if minimum is not None and value < minimum:
        raise ValidationError(
            f"{name} is below its minimum of {minimum}. Set it to at least "
            f"{minimum}, or leave it unset for the default."
        )
    if maximum is not None and value > maximum:
        raise ValidationError(
            f"{name} is above its maximum of {maximum}. Set it to at most "
            f"{maximum}, or leave it unset for the default."
        )
    return value


__all__ = [
    "FALSE_WORDS",
    "TRUE_WORDS",
    "env_flag",
    "env_int",
    "env_path",
    "env_paths",
    "env_str",
]
