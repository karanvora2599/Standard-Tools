"""
The OPT-IN reference: a tool that answers in scalars and can also leave the
whole series behind.

Several tools here compute a per-bar series, summarize it into a handful of
numbers and drop the series on the floor. The summary is usually the answer,
so the series must not become mandatory output -- inlining hundreds of floats
beside the six numbers that describe them is how a payload stops being
readable. But when the series IS the question (charting a hedge ratio's
drift, scoring a regime label sequence, feeding a rolling Sharpe to another
tool), there was no way to obtain it at all.

So the tool takes `run_id` and `name`, both optional. Give both and the
series is published and the reference comes back; give neither and the
result is exactly what it was. Give ONE and the call is refused, because
half of an address is not an address -- `sqt://<kind>/<run_id>/<name>`
needs both halves, and silently ignoring the half that was given would
answer without the reference the caller asked for.

Publishing goes through `handoff.publish` with its default
`overwrite=False`, so a reused `(run_id, name)` fails loudly rather than
replacing a value some other holder's reference already promises.

See the CHANGELOG entry of 2026-09-22.
"""

from __future__ import annotations

from typing import Any, Optional

from standard_quant_tools.error import ValidationError

from .handoff import publish

__all__ = ["publish_if_requested"]


def publish_if_requested(
    data: Any,
    *,
    kind: str,
    run_id: Optional[str],
    name: Optional[str],
    producer: str,
) -> Optional[str]:
    """
    The reference, `None` when none was asked for, or a refusal naming what
    is missing.

    `producer` is both the recorded producer and the name the refusal uses,
    so the message says which tool would not publish.
    """
    if run_id is None and name is None:
        return None
    if run_id is None or name is None:
        given, missing = ("name", "run_id") if run_id is None else ("run_id", "name")
        raise ValidationError(
            f"{producer}: {given} was given without {missing}. A reference is "
            f"addressed by BOTH -- sqt://{kind}/<run_id>/<name> -- so half of "
            f"one addresses nothing, and publishing under an invented "
            f"{missing} would hand back a reference the caller cannot "
            f"predict. Pass {missing} as well, or pass neither and read the "
            "summary inline."
        )
    return publish(
        data,
        kind=kind,
        run_id=run_id,
        name=name,
        producer=producer,
    )
