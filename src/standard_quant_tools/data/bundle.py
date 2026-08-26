"""
A DataBundle: several frames, each carrying the contract it was fetched
under.

WHY A CONTAINER AT ALL. A dataset is rarely one frame. Bars and the
fundamentals joined onto them come from different places, on different
calendars, under different guarantees — and the guarantee is the part that
gets lost. A frame passed around on its own is just columns; whether its
values were knowable when they claim to be is a fact about where it came
from, and that fact lives in the fetch call rather than in the data.

So a bundle pairs every frame with its `TemporalContract`, and the pairing
is what makes `validate_bundle` able to answer "is this safe to model on"
without re-deriving anything.

WHAT THIS DELIBERATELY DOES NOT HOLD. The expansion plan listed eleven slots
— `bars`, `trades`, `quotes`, `orderbook`, `fundamentals`, `estimates`,
`macro`, `events`, `options`, `reference_data`, `corporate_actions`. Eight
of those have no provider behind them in this library, and a container with
eight permanently empty slots is the "empty box with a correct label" that
`point_in_time.py` already warns about. It also actively invites the wrong
thing: a slot named `fundamentals` reads as an invitation to fill it from a
source with no availability timestamps, which is the exact leak the contract
exists to stop.

A bundle therefore holds whatever frames it was GIVEN, keyed by frame kind.
Adding `fundamentals` needs no change here — it needs a source.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from standard_quant_tools.data.temporal import (
    FRAME_KINDS,
    TemporalContract,
    contract_for_frame,
)
from standard_quant_tools.error import ValidationError


class DataBundle:
    """
    Frames plus the contract each was fetched under.

    Not a Pydantic model: the values are DataFrames, which do not belong in
    a JSON schema. What crosses a tool boundary is `describe()`, which is
    the metadata; the frames themselves cross as `sqt://` references like
    everything else bulk.
    """

    def __init__(self, name: str = "bundle") -> None:
        self.name = name
        self._frames: Dict[str, Any] = {}
        self._contracts: Dict[str, TemporalContract] = {}

    # ── building ────────────────────────────────────────────────────────

    def add(
        self,
        frame_kind: str,
        frame,
        contract: Optional[TemporalContract] = None,
        *,
        source: str = "unknown",
        entity_scoped: bool = True,
    ) -> "DataBundle":
        """
        Add one frame under its kind.

        `contract` is optional and inferred from the frame when omitted --
        but inference reads COLUMNS, which can only ever tell you what is
        there and not what the provider guarantees. Passing the provider's
        own contract is better whenever you have it, and the difference
        shows up in `revisions`: a frame that happens to contain no
        restatement is indistinguishable from one whose source discards
        them, and inference correctly refuses to guess between those.
        """
        if frame_kind not in FRAME_KINDS:
            raise ValidationError(
                f"unknown frame kind {frame_kind!r}; expected one of "
                f"{list(FRAME_KINDS)}"
            )
        if frame_kind in self._frames:
            raise ValidationError(
                f"{self.name}: {frame_kind!r} is already in this bundle. "
                "Replacing it silently would mean two callers disagreeing "
                "about which frame is the one that was validated."
            )
        self._frames[frame_kind] = frame
        self._contracts[frame_kind] = contract or contract_for_frame(
            frame,
            source=source,
            frame_kind=frame_kind,
            entity_scoped=entity_scoped,
        )
        return self

    # ── reading ─────────────────────────────────────────────────────────

    def __contains__(self, frame_kind: str) -> bool:
        return frame_kind in self._frames

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def kinds(self) -> List[str]:
        return sorted(self._frames)

    def contract(self, frame_kind: str) -> TemporalContract:
        self._require(frame_kind)
        return self._contracts[frame_kind]

    def frame(self, frame_kind: str, *, require_pit: bool = False):
        """
        The frame, optionally gated on its contract.

        `require_pit=True` is the point of the whole type. A caller about to
        join a frame onto a modeling panel asks for it that way, and a frame
        whose source cannot say when its values became knowable is refused
        HERE -- before the join, before the fit, and with the provider named
        rather than with a column-missing error three frames down.
        """
        self._require(frame_kind)
        if require_pit:
            from standard_quant_tools.data.temporal import require_pit as _require_pit

            _require_pit(
                self._contracts[frame_kind],
                f"using {frame_kind!r} from {self.name!r}",
            )
        return self._frames[frame_kind]

    def _require(self, frame_kind: str) -> None:
        if frame_kind not in self._frames:
            raise ValidationError(
                f"{self.name} has no {frame_kind!r} frame. It holds: "
                f"{self.kinds or '(nothing)'}"
            )

    # ── describing ──────────────────────────────────────────────────────

    def describe(self) -> Dict[str, Any]:
        """Everything about this bundle that survives a JSON boundary."""
        frames = []
        for kind in self.kinds:
            frame = self._frames[kind]
            contract = self._contracts[kind]
            frames.append(
                {
                    "frame_kind": kind,
                    "source": contract.source,
                    "rows": int(len(frame)) if hasattr(frame, "__len__") else None,
                    "columns": sorted(getattr(frame, "columns", [])),
                    "pit_safe": contract.pit_safe,
                    "reproduces_history": contract.reproduces_history,
                    "revisions": contract.revisions,
                    "caveats": contract.caveats(),
                }
            )
        return {
            "name": self.name,
            "n_frames": len(self._frames),
            "kinds": self.kinds,
            "frames": frames,
            "pit_safe": self.pit_safe,
            "reproduces_history": self.reproduces_history,
            "warnings": self.warnings(),
        }

    @property
    def pit_safe(self) -> bool:
        """
        Whether EVERY frame in the bundle can be joined point-in-time.

        The weakest link, not the average. A bundle is used as a unit, and
        one frame that cannot say when its values became knowable
        contaminates whatever is built from the whole thing -- reporting
        "mostly safe" would be reporting a number nobody can act on.
        """
        return all(c.pit_safe for c in self._contracts.values())

    @property
    def reproduces_history(self) -> bool:
        return all(c.reproduces_history for c in self._contracts.values())

    def warnings(self) -> List[str]:
        """Every frame's caveats, attributed to the frame they came from."""
        out: List[str] = []
        for kind in self.kinds:
            for caveat in self._contracts[kind].caveats():
                out.append(f"[{kind}] {caveat}")
        return out


def validate_bundle(bundle: DataBundle, *, require_pit: bool = True) -> Dict[str, Any]:
    """
    Is this bundle safe to model on, and what is wrong with it if not.

    Returns a verdict rather than raising, because the answer is usually
    "yes, with caveats" and a caller needs to see the caveats to decide.
    `require_pit=False` is for a caller who genuinely does not need a
    point-in-time join -- a descriptive study of the present, say -- and it
    is deliberately a decision they have to make in writing.
    """
    if len(bundle) == 0:
        raise ValidationError(
            f"{bundle.name}: nothing to validate -- the bundle is empty."
        )

    blocking: List[str] = []
    if require_pit:
        for kind in bundle.kinds:
            reason = bundle.contract(kind).why_not_pit_safe()
            if reason:
                blocking.append(f"[{kind}] {reason}")

    return {
        "name": bundle.name,
        "n_frames": len(bundle),
        "kinds": bundle.kinds,
        "pit_safe": bundle.pit_safe,
        "reproduces_history": bundle.reproduces_history,
        "usable": not blocking,
        "blocking": blocking,
        "warnings": bundle.warnings(),
    }


__all__ = ["DataBundle", "validate_bundle"]
