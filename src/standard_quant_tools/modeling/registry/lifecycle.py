"""
Lifecycle stages: where a model is on the way from fitted to trusted.

WHY A LOG AND NOT A FIELD. The manifest is content-hashed and immutable --
that is the property every integrity check here rests on -- so a stage
cannot be a field on it without either rewriting the commit point or
signing the stage into the hashes it would then be changing. A stage is a
DECISION made after registration, by someone, for a reason, on evidence,
and possibly reversed. That is a record with a history, not a value: it
lives in `promotions.jsonl` beside the manifest, append-only, and the
current stage is whatever the last line says. Nothing about the model's
identity changes when it is promoted, and nothing about its promotion
history can be edited without the edit showing.

THE STAGES. `candidate` is what registration produces: a model that has
been fitted and walk-forward validated, which is a fact about the fit and
not a judgement about the evidence. `validated` says someone read the
evidence and accepted it. `staging` and `production` are deployment
states. `archived` is terminal. A promotion moves one stage forward at a
time -- skipping `validated` on the way to `production` is the decision
this exists to make visible -- while a demotion may go back to any earlier
live stage, because rolling back is also a decision worth recording.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Sequence

from standard_quant_tools.error import ValidationError

from .. import artifacts as _artifacts
from . import mirror as _mirror

STAGES = ("candidate", "validated", "staging", "production", "archived")
LifecycleStage = Literal["candidate", "validated", "staging", "production", "archived"]
_ORDER = {stage: position for position, stage in enumerate(STAGES)}
PROMOTIONS_FILE = "promotions.jsonl"
INITIAL_STAGE = "candidate"


@dataclass(frozen=True)
class Promotion:
    """One line of the log: a decision, its reason and what it rested on."""

    from_stage: str
    to_stage: str
    reason: str
    actor: str
    timestamp_utc: str
    evidence: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _log_path(model_id: str):
    directory = _artifacts.run_dir(model_id)
    if not (directory / "manifest.json").exists():
        raise ValidationError(f"no registered model with model_id={model_id!r}")
    return directory / PROMOTIONS_FILE


def promotions(model_id: str) -> List[Promotion]:
    """Every promotion recorded for the model, oldest first; empty for a
    model that was never promoted, which is every model at registration."""
    path = _log_path(model_id)
    if not path.exists():
        return []
    out: List[Promotion] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            out.append(
                Promotion(
                    from_stage=str(record["from_stage"]),
                    to_stage=str(record["to_stage"]),
                    reason=str(record["reason"]),
                    actor=str(record.get("actor", "unknown")),
                    timestamp_utc=str(record["timestamp_utc"]),
                    evidence=[str(e) for e in record.get("evidence") or []],
                )
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise ValidationError(
                f"{path.name} line {number} for model {model_id!r} is not a "
                f"promotion record: {exc}. The log is append-only; a line that "
                "cannot be read means the file was edited, and the stage "
                "cannot be trusted until it is repaired."
            ) from exc
    return out


def current_stage(model_id: str) -> str:
    """The stage the last promotion moved the model to, or `candidate`."""
    history = promotions(model_id)
    return history[-1].to_stage if history else INITIAL_STAGE


def promote(
    model_id: str,
    to_stage: str,
    reason: str,
    *,
    actor: str = "agent",
    evidence: Sequence[str] = (),
) -> Promotion:
    """
    Record a stage change, after checking it is one the lifecycle allows.

    Refused by name: an unknown stage, a promotion to the stage the model is
    already at, any move out of `archived`, a forward move that skips a
    stage, and a reason too short to be one.
    """
    if to_stage not in STAGES:
        raise ValidationError(
            f"promote_model: {to_stage!r} is not a lifecycle stage; the stages "
            f"are {list(STAGES)}, in that order."
        )
    if not reason or len(reason.strip()) < 8:
        raise ValidationError(
            "promote_model: a promotion needs a reason of at least a few "
            "words. It is read months later by someone deciding whether to "
            "trust the model, and 'ok' does not help them."
        )
    stage = current_stage(model_id)
    if stage == "archived":
        raise ValidationError(
            f"promote_model: model {model_id!r} is archived, which is "
            "terminal. Retrain to get a new model_id rather than reviving "
            "one whose history says it was retired."
        )
    if to_stage == stage:
        raise ValidationError(
            f"promote_model: model {model_id!r} is already at {stage!r}."
        )
    if to_stage != "archived" and _ORDER[to_stage] > _ORDER[stage] + 1:
        skipped = STAGES[_ORDER[stage] + 1 : _ORDER[to_stage]]
        raise ValidationError(
            f"promote_model: {stage!r} -> {to_stage!r} skips {list(skipped)}. "
            "A model reaches production one stage at a time, so every step "
            "is a decision somebody made and recorded; promote it to "
            f"{skipped[0]!r} first."
        )
    record = Promotion(
        from_stage=stage,
        to_stage=to_stage,
        reason=reason.strip(),
        actor=str(actor or "agent"),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        evidence=[str(e) for e in evidence],
    )
    path = _log_path(model_id)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    # The log is the stage; a mirror holding a stale log holds a stale
    # stage, so the whole file follows every decision.
    _mirror.mirror_file(model_id, PROMOTIONS_FILE)
    return record


__all__ = [
    "INITIAL_STAGE",
    "PROMOTIONS_FILE",
    "STAGES",
    "Promotion",
    "current_stage",
    "promote",
    "promotions",
]
