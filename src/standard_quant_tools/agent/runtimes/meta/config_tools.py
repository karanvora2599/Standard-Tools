"""
What this process is actually configured to do.

Twenty `SQT_*` environment variables govern whether decisions are recorded,
where artifacts land, which provider can be reached, whether the compiled
extension is used and how a model package is signed -- and until now not
one of them was reported by any tool. An agent learned its own
configuration by triggering it: a run whose artifacts vanished, a fetch
that refused for want of a credential, a decision log that recorded
nothing. That is the same enforced-but-unreadable shape the numeric
contract had, one layer further out.

TWO RULES MAKE THIS SAFE AND HONEST.

A SECRET IS REPORTED AS SET, NEVER AS A VALUE. The redaction salt is the
sharp case: it exists to stop an offline brute force of the redaction
placeholders, so printing it would hand back exactly what redaction was
protecting. The same holds for a private signing key path, an API key, a
bearer token and a mirror URL, which may carry credentials in its userinfo.
`set` answers the only question a caller legitimately has -- is this
configured -- and the value stays where it was put.

THE VALUE IS THE EFFECTIVE ONE. Every non-secret is resolved through the
function the library itself reads it with, not through `os.environ`. That
difference is the whole point: an unset `SQT_AUDIT_DIR` still has a
concrete answer, `SQT_CACHE_DIR` is frozen at import so a later change to
the environment does not move the cache, and `SQT_AUDIT_ENABLED=""` reads
as OFF rather than as unset. A tool that echoed the raw environment would
report a configuration nobody is running.

The audit settings are declared here once and read from here by
`describe_audit_log`, so the two tools cannot come to disagree about what
the trail is configured to do.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class _Result(BaseModel):
    model_config = ConfigDict(extra="allow")

    warnings: List[str] = Field(default_factory=list)


class EffectiveConfigInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_paths: bool = Field(
        True,
        description=(
            "Also report the platform directory variables the audit path "
            "consults when SQT_AUDIT_DIR is unset. They are not this "
            "library's own settings, which is why they are separable, but "
            "they decide where the decision log lives."
        ),
    )


class ConfigSetting(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = ""
    category: str = ""
    value: Optional[str] = Field(
        None,
        description=(
            "The EFFECTIVE value, resolved through the function that reads "
            "it rather than echoed from the environment -- so an unset "
            "variable still reports the answer the library will use. Always "
            "null for a secret."
        ),
    )
    set: bool = Field(
        False,
        description="Whether the variable is present in this environment.",
    )
    default: Optional[str] = Field(
        None, description="What the library uses when it is not set."
    )
    is_secret: bool = Field(
        False,
        description=(
            "True when the value is withheld on purpose. Reporting it would "
            "disclose a credential, a private key location or the redaction "
            "salt, which exists precisely to make the placeholders it "
            "produces unrecoverable."
        ),
    )
    reader: str = Field("", description="The function in this library that reads it.")
    effect: str = Field("", description="What changes when it is set.")


class EffectiveConfigResult(_Result):
    n_settings: int = 0
    n_set: int = 0
    n_secrets_set: int = 0
    settings: List[ConfigSetting] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _Setting:
    """One environment variable, with the library's own way of reading it."""

    name: str
    category: str
    reader: str
    default: Optional[str]
    is_secret: bool
    effect: str
    #: Returns the effective value as a string, or None when there is none.
    #: Never called for a secret.
    resolve: Optional[Callable[[], Optional[str]]] = None


def _yes_no(value: bool) -> str:
    return "true" if value else "false"


def _audit_enabled_value() -> Optional[str]:
    from standard_quant_tools.audit.paths import _audit_enabled

    return _yes_no(_audit_enabled())


def _audit_dir_value() -> Optional[str]:
    from standard_quant_tools.audit.paths import _audit_dir

    return str(_audit_dir())


def _fail_closed_value() -> Optional[str]:
    from standard_quant_tools.audit.dispatch import _audit_fail_closed

    return _yes_no(_audit_fail_closed())


def _redact_fields_value() -> Optional[str]:
    from standard_quant_tools.audit.redaction import _redact_fields

    fields = _redact_fields()
    return ",".join(fields) if fields else None


def _retention_days_value() -> Optional[str]:
    from standard_quant_tools.audit.retention import _retention_days_from_env

    days = _retention_days_from_env()
    return None if days is None else str(days)


def _runs_dir_value() -> Optional[str]:
    from standard_quant_tools._runspath import runs_dir

    return str(runs_dir())


def _cache_dir_value() -> Optional[str]:
    from standard_quant_tools.data import _cache

    return str(_cache._CACHE_ROOT)


def _native_disabled_value() -> Optional[str]:
    from standard_quant_tools import native_disabled

    return _yes_no(native_disabled())


def _bloomberg_host_value() -> Optional[str]:
    from standard_quant_tools.data.bloomberg_provider import (
        _resolve_bloomberg_config,
    )

    return str(_resolve_bloomberg_config()[0])


def _bloomberg_port_value() -> Optional[str]:
    from standard_quant_tools.data.bloomberg_provider import (
        _resolve_bloomberg_config,
    )

    return str(_resolve_bloomberg_config()[1])


def _model_format_value() -> Optional[str]:
    from standard_quant_tools.modeling.registry.serialization import default_format

    return str(default_format())


def _verify_key_value() -> Optional[str]:
    from standard_quant_tools.modeling.registry.signing import VERIFY_KEY_ENV

    return os.environ.get(VERIFY_KEY_ENV) or None


def _fetch_concurrency_value() -> Optional[str]:
    from standard_quant_tools.modeling.dataset.fetch import _max_concurrency

    return str(_max_concurrency())


def _num_threads_value() -> Optional[str]:
    from standard_quant_tools.modeling.registry.environment import THREAD_VARIABLES

    # The same read the run fingerprint makes, through the module that owns
    # the list -- so a variable dropped from the fingerprint stops being
    # reported here too, rather than being described as recorded when it
    # is not.
    if "SQT_NUM_THREADS" not in THREAD_VARIABLES:  # pragma: no cover - guard
        return None
    return os.environ.get("SQT_NUM_THREADS") or None


def _local_app_data_value() -> Optional[str]:
    return os.environ.get("LOCALAPPDATA") or None


def _xdg_state_home_value() -> Optional[str]:
    return os.environ.get("XDG_STATE_HOME") or None


#: The seven settings that govern the decision log. Declared as their own
#: group because `describe_audit_log` reports them too, and two tools
#: maintaining two copies of this table is exactly how they would come to
#: disagree about whether recording is on.
AUDIT_SETTINGS: Tuple[_Setting, ...] = (
    _Setting(
        name="SQT_AUDIT_ENABLED",
        category="audit",
        reader="audit.paths._audit_enabled",
        default="1 (recording on)",
        is_secret=False,
        effect=(
            "Whether dispatch() writes a decision record at all. Off, the "
            "trail stays empty and no call is recoverable afterwards."
        ),
        resolve=_audit_enabled_value,
    ),
    _Setting(
        name="SQT_AUDIT_DIR",
        category="audit",
        reader="audit.paths._audit_dir",
        default="the platform state directory, under standard_quant_tools/audit",
        is_secret=False,
        effect=(
            "Where the decision log lives. Point it somewhere backed up: "
            "the default is state, not cache, precisely because the trail "
            "is the file you cannot recreate."
        ),
        resolve=_audit_dir_value,
    ),
    _Setting(
        name="SQT_AUDIT_FAIL_CLOSED",
        category="audit",
        reader="audit.dispatch._audit_fail_closed",
        default="0 (the result is returned even if the record fails)",
        is_secret=False,
        effect=(
            "Whether a failed record write fails the call. On, an action "
            "taken without a record of it does not reach the caller."
        ),
        resolve=_fail_closed_value,
    ),
    _Setting(
        name="SQT_AUDIT_REDACT_FIELDS",
        category="audit",
        reader="audit.redaction._redact_fields",
        default=None,
        is_secret=False,
        effect=(
            "Comma-separated dotted paths into a recorded input that are "
            "replaced by a hashed placeholder. Which fields are redacted is "
            "reportable; what they contained is not."
        ),
        resolve=_redact_fields_value,
    ),
    _Setting(
        name="SQT_AUDIT_REDACT_SALT",
        category="audit",
        reader="audit.redaction._placeholder_for",
        default=None,
        is_secret=True,
        effect=(
            "Mixes a secret into every redaction placeholder. Unsalted, a "
            "placeholder over a small value space (an account id, a PIN) is "
            "brute-forceable offline, which is the whole gap this closes -- "
            "so the salt itself is never reported."
        ),
    ),
    _Setting(
        name="SQT_AUDIT_RETENTION_DAYS",
        category="audit",
        reader="audit.retention._retention_days_from_env",
        default=None,
        is_secret=False,
        effect=(
            "The window past which a day file becomes a deletion CANDIDATE. "
            "Unset means nothing is ever a candidate, not that everything "
            "is. Deletion itself is an operator action with a CLI."
        ),
        resolve=_retention_days_value,
    ),
    _Setting(
        name="SQT_AUDIT_SIGNING_KEY_PATH",
        category="audit",
        reader="audit.signing._load_signer",
        default=None,
        is_secret=True,
        effect=(
            "The Ed25519 private key a day's checkpoint is signed with. A "
            "signed checkpoint is the only check that catches a wholesale "
            "rewrite, which the hash chain can recompute."
        ),
    ),
)


_OTHER_SETTINGS: Tuple[_Setting, ...] = (
    _Setting(
        name="SQT_RUNS_DIR",
        category="storage",
        reader="_runspath.runs_dir",
        default="a cache directory under the home directory",
        is_secret=False,
        effect=(
            "Where every artifact and every handoff reference resolves. Two "
            "processes pointed at different roots cannot read each other's "
            "references."
        ),
        resolve=_runs_dir_value,
    ),
    _Setting(
        name="SQT_CACHE_DIR",
        category="storage",
        reader="data._cache._CACHE_ROOT",
        default="a cache directory under the home directory",
        is_secret=False,
        effect=(
            "The Parquet bar cache. Read ONCE at import, so changing it "
            "after this process started does not move the cache -- which is "
            "why the effective value is reported rather than the variable."
        ),
        resolve=_cache_dir_value,
    ),
    _Setting(
        name="SQT_DISABLE_NATIVE",
        category="execution",
        reader="standard_quant_tools.native_disabled",
        default="0 (the compiled extension is used when present)",
        is_secret=False,
        effect=(
            "Makes the compiled extension unimportable, so every kernel "
            "takes its fallback path. Which path ran is recorded per call."
        ),
        resolve=_native_disabled_value,
    ),
    _Setting(
        name="SQT_MCP_TOKEN",
        category="server",
        reader="mcp.config._resolve_transport",
        default=None,
        is_secret=True,
        effect=(
            "The bearer token the HTTP server requires. Read from the "
            "environment and never from a flag, because a command line is "
            "visible to every user on the box."
        ),
    ),
    _Setting(
        name="SQT_POLYGON_API_KEY",
        category="provider",
        reader="data.polygon_provider._resolve_polygon_api_key",
        default=None,
        is_secret=True,
        effect=(
            "The credential the point-in-time provider needs. Without it "
            "that provider is reported unavailable rather than failing "
            "mid-fetch."
        ),
    ),
    _Setting(
        name="SQT_BLOOMBERG_HOST",
        category="provider",
        reader="data.bloomberg_provider._resolve_bloomberg_config",
        default="localhost",
        is_secret=False,
        effect="The Desktop API host the Bloomberg provider connects to.",
        resolve=_bloomberg_host_value,
    ),
    _Setting(
        name="SQT_BLOOMBERG_PORT",
        category="provider",
        reader="data.bloomberg_provider._resolve_bloomberg_config",
        default="8194",
        is_secret=False,
        effect="The Desktop API port. A non-integer value is refused by name.",
        resolve=_bloomberg_port_value,
    ),
    _Setting(
        name="SQT_MODEL_FORMAT",
        category="modeling",
        reader="modeling.registry.serialization.default_format",
        default="joblib",
        is_secret=False,
        effect=(
            "Which serialization a registered model is loaded from when the "
            "caller does not say. An unrecognised value is refused by name "
            "rather than silently taking the default."
        ),
        resolve=_model_format_value,
    ),
    _Setting(
        name="SQT_MODEL_MIRROR_URL",
        category="modeling",
        reader="modeling.registry.mirror.configured_mirror",
        default=None,
        is_secret=True,
        effect=(
            "An artifact store every registration is pushed to. Withheld "
            "because a URL can carry credentials in its userinfo, so the "
            "value is not safely printable even though the setting is."
        ),
    ),
    _Setting(
        name="SQT_MODEL_SIGNING_KEY_PATH",
        category="modeling",
        reader="modeling.registry.signing.signing_configured",
        default=None,
        is_secret=True,
        effect=(
            "The private key a model manifest is signed with at "
            "registration. Its presence is what makes a later verification "
            "meaningful."
        ),
    ),
    _Setting(
        name="SQT_MODEL_VERIFY_KEY_PATH",
        category="modeling",
        reader="modeling.registry.signing._public_key_bytes",
        default=None,
        is_secret=False,
        effect=(
            "The PUBLIC key a manifest is verified against when no key is "
            "passed. Reportable exactly because it is public -- pinning it "
            "is a statement a caller should be able to read back."
        ),
        resolve=_verify_key_value,
    ),
    _Setting(
        name="SQT_MODELING_FETCH_CONCURRENCY",
        category="modeling",
        reader="modeling.dataset.fetch._max_concurrency",
        default="8",
        is_secret=False,
        effect=(
            "How many symbols a dataset build fetches at once. An unusable "
            "value falls back to the default rather than failing the fetch."
        ),
        resolve=_fetch_concurrency_value,
    ),
    _Setting(
        name="SQT_NUM_THREADS",
        category="execution",
        reader="modeling.registry.environment.environment_fingerprint",
        default=None,
        is_secret=False,
        effect=(
            "A thread cap recorded in a run's environment fingerprint. "
            "Unset is recorded as unset: 'no cap was set' is a different "
            "environment from 'capped at one'."
        ),
        resolve=_num_threads_value,
    ),
)

#: The twenty settings this library reads, in one place.
SETTINGS: Tuple[_Setting, ...] = AUDIT_SETTINGS + _OTHER_SETTINGS

#: Consulted by the audit path when SQT_AUDIT_DIR is unset. Not this
#: library's own variables, which is why they are behind a flag, but they
#: decide where the decision log ends up.
_PATH_SETTINGS: Tuple[_Setting, ...] = (
    _Setting(
        name="LOCALAPPDATA",
        category="path",
        reader="audit.paths._audit_dir",
        default="the local application data directory under the home directory",
        is_secret=False,
        effect=(
            "On Windows, the root the decision log is placed under when "
            "SQT_AUDIT_DIR is unset."
        ),
        resolve=_local_app_data_value,
    ),
    _Setting(
        name="XDG_STATE_HOME",
        category="path",
        reader="audit.paths._audit_dir",
        default="~/.local/state",
        is_secret=False,
        effect=(
            "Elsewhere, the root the decision log is placed under when "
            "SQT_AUDIT_DIR is unset. State rather than cache, because a "
            "cache is the directory a user is invited to delete."
        ),
        resolve=_xdg_state_home_value,
    ),
)


def resolve_setting(setting: _Setting) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    One setting as `(value, set, problem)`.

    `problem` is a sentence when the library's own reader REFUSED the value
    -- a port that is not an integer, a model format that is not a format.
    That refusal is the most useful thing this tool can report about a
    variable, so it is carried as a warning rather than raised: a broken
    setting must not make the report that would explain it unavailable.
    """
    present = setting.name in os.environ
    if setting.is_secret or setting.resolve is None:
        return None, present, None
    try:
        return setting.resolve(), present, None
    except Exception as exc:  # noqa: BLE001 - the refusal IS the finding
        logger.debug("[describe_effective_config] %s", setting.name, exc_info=True)
        return (
            None,
            present,
            f"{setting.name} could not be resolved: {exc}. The library "
            "refuses the value it is set to, so whatever reads it next "
            "will refuse in the same way.",
        )


def describe_effective_config(
    input_data: EffectiveConfigInput,
) -> EffectiveConfigResult:
    """
    Every `SQT_*` setting this process reads, and what it resolves to.

    Configuration decides whether a decision is recorded, where an artifact
    lands and which provider can be reached, and none of it was reachable
    from a tool -- so an agent discovered its own configuration by tripping
    over it. The values here are the EFFECTIVE ones, taken through the
    functions the library reads them with, which is why an unset variable
    still has an answer.

    A secret reports only whether it is set. The redaction salt is the
    reason the rule is absolute rather than a judgement call: the salt
    exists to make a redaction placeholder unrecoverable, so disclosing it
    would undo the redaction it configures.

    Fetches nothing, writes nothing, and changes no setting -- this reads
    configuration and cannot alter it.
    """
    from standard_quant_tools.config import load_env

    # The same first step every reader in the library takes, so a value
    # supplied by a local .env file is reported as the process will see it
    # rather than as absent.
    try:
        load_env()
    except Exception:  # noqa: BLE001 - a missing .env is the normal state
        logger.debug("[describe_effective_config] load_env failed", exc_info=True)

    wanted: List[_Setting] = list(SETTINGS)
    if input_data.include_paths:
        wanted += list(_PATH_SETTINGS)

    warnings: List[str] = []
    rows: List[ConfigSetting] = []
    for setting in wanted:
        value, present, problem = resolve_setting(setting)
        if problem:
            warnings.append(problem)
        rows.append(
            ConfigSetting(
                name=setting.name,
                category=setting.category,
                value=value,
                set=present,
                default=setting.default,
                is_secret=setting.is_secret,
                reader=setting.reader,
                effect=setting.effect,
            )
        )

    n_set = sum(1 for row in rows if row.set)
    n_secrets_set = sum(1 for row in rows if row.set and row.is_secret)

    notes: List[str] = [
        "`value` is what the library RESOLVED, not what the environment "
        "holds: an unset variable still reports the default in force, and a "
        "value read once at import reports what is in force now rather than "
        "what the environment says today.",
        "A secret reports `set` and nothing else. That is the whole answer "
        "to 'is this configured'; the value stays where it was put.",
    ]
    if not input_data.include_paths:
        notes.append(
            "The platform directory variables the audit path falls back to "
            "were omitted. Set include_paths to see where the decision log "
            "lands when SQT_AUDIT_DIR is unset."
        )

    configured = {row.name for row in rows if row.set}
    if "SQT_AUDIT_DIR" not in configured:
        warnings.append(
            "SQT_AUDIT_DIR is unset, so the decision log is under a "
            "platform default that a machine rebuild or a container restart "
            "does not preserve. Point it at durable storage if the trail is "
            "meant to outlive this host."
        )
    if "SQT_AUDIT_REDACT_FIELDS" in configured and (
        "SQT_AUDIT_REDACT_SALT" not in configured
    ):
        warnings.append(
            "Fields are being redacted with no SQT_AUDIT_REDACT_SALT, so the "
            "placeholders are unsalted and brute-forceable offline for any "
            "field with a small value space. Set a salt and keep it stable."
        )
    if "SQT_MODEL_MIRROR_URL" in configured:
        warnings.append(
            "A model mirror is configured, so every registration in this "
            "process is pushed to a second store. The URL is withheld "
            "because it can carry credentials."
        )

    logger.debug("[describe_effective_config] %d settings, %d set", len(rows), n_set)
    return EffectiveConfigResult(
        n_settings=len(rows),
        n_set=n_set,
        n_secrets_set=n_secrets_set,
        settings=rows,
        notes=notes,
        warnings=warnings,
    )


__all__ = [
    "AUDIT_SETTINGS",
    "SETTINGS",
    "ConfigSetting",
    "EffectiveConfigInput",
    "EffectiveConfigResult",
    "describe_effective_config",
    "resolve_setting",
]
