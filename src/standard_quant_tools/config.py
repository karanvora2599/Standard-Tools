"""
Single choke point for loading provider configuration/secrets from a local
`.env` file, so credentials never need to be hardcoded or passed as CLI
args. In CI (GitHub Actions / GitLab CI), the same environment variables are
instead injected directly by the platform's secrets mechanism — `load_env()`
is a no-op in that case (there is no `.env` file to find, and real env vars
that are already set are never overridden), so provider code that reads
`os.environ` works identically in both places.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_loaded = False


def load_env(dotenv_path: Optional[str] = None) -> bool:
    """
    Load key=value pairs from a `.env` file into `os.environ`, if present.

    Idempotent per process (subsequent calls are no-ops) and silent by
    design when there's nothing to load — a missing `.env` is the expected,
    normal state in CI or any environment where secrets already arrive as
    real environment variables. Existing environment variables are never
    overridden (`override=False`), so a real secret injected by CI always
    wins over a stale value left in a local `.env`.

    Args:
        dotenv_path: Explicit path to a `.env` file. Defaults to `.env` in
            the current working directory (python-dotenv's own default),
            which is the repo root for any normal invocation of this
            package's code, tests, or the `sqt` CLI.

    Returns:
        True if a file was found and loaded, False otherwise (including
        when this function has already run once in this process, or when
        `python-dotenv` isn't installed).
    """
    global _loaded
    if _loaded:
        return False

    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug(
            "[config] python-dotenv not installed — skipping .env load; "
            "environment variables set another way (shell, CI secrets) "
            "still work normally."
        )
        _loaded = True
        return False

    kwargs: Dict[str, Any] = {"override": False}
    if dotenv_path is not None:
        kwargs["dotenv_path"] = dotenv_path
    found = load_dotenv(**kwargs)
    _loaded = True
    if found:
        logger.debug("[config] loaded .env from %s", dotenv_path or Path.cwd() / ".env")
    return found


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Convenience wrapper: ensure .env is loaded, then read one variable."""
    load_env()
    return os.environ.get(name, default)
