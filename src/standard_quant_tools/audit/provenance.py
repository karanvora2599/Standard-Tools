"""Best-effort reproducibility provenance: C++ extension availability, the
current git commit, the installed package version, and a content hash of a
registered strategy's source code. All of these fail silently (return
`None`/`False`) rather than raise — provenance is a nice-to-have, never a
reason to break a tool call."""

from pathlib import Path
from typing import Any, Optional

from .hashing import hash_payload


def _cpp_available() -> bool:
    try:
        import standard_quant_tools._sqt_core  # type: ignore[attr-defined]  # noqa: F401

        return True
    except ImportError:
        return False


_git_sha_cache: Optional[str] = None
_git_sha_resolved = False


def _git_sha() -> Optional[str]:
    """
    Best-effort `git rev-parse HEAD` in the repo containing this package.
    Returns None (never raises) outside a git checkout, without git
    installed, or in any other failure mode. Resolved once per process and
    cached.
    """
    global _git_sha_cache, _git_sha_resolved
    if _git_sha_resolved:
        return _git_sha_cache
    _git_sha_resolved = True
    try:
        import subprocess

        # parents[0]=audit, [1]=standard_quant_tools, [2]=src, [3]=repo root.
        repo_root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            _git_sha_cache = result.stdout.strip() or None
    except Exception:
        _git_sha_cache = None
    return _git_sha_cache


def _package_version() -> Optional[str]:
    try:
        from standard_quant_tools import __version__

        return __version__
    except Exception:
        return None


def _strategy_source_hash(model_instance: Any) -> Optional[str]:
    """
    Content hash of a registered strategy's source code, when
    `model_instance` names one via a `strategy` or `strategy_type` field
    (e.g. WalkForwardInput.strategy, BacktestDiagnosticsInput.strategy_type).
    None when neither field is present, the name isn't a registered
    strategy (e.g. a custom-signal tool), or on any lookup failure.
    """
    strategy_name = getattr(model_instance, "strategy", None) or getattr(
        model_instance, "strategy_type", None
    )
    if not strategy_name:
        return None
    try:
        import inspect

        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        fn = STRATEGY_REGISTRY.get(strategy_name)
        if fn is None:
            return None
        return hash_payload(inspect.getsource(fn))
    except Exception:
        return None
