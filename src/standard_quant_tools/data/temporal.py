"""
The temporal contract: what a data source can tell you about WHEN.

Every non-price dataset carries a leak waiting to happen. A quarterly filing
describes 30 September and is published on 25 October; a model that joins it
on 30 September has three weeks of hindsight in every row, and the backtest
that results looks like skill. `modeling/dataset/point_in_time.py` already
has the join that avoids this and already refuses a frame that cannot supply
`available_time`.

What was missing is the DECLARATION. The refusal fires at join time, which
is late: an agent that has already fetched, cleaned and cached a fundamental
history finds out at the last step that the source could never have
supported the thing it was building. This module makes the same question
answerable first, in one call, without fetching anything.

TWO TIMESTAMPS, NOT THREE. The expansion plan called for
`event_time` / `available_at` / `revision_time`. Measured against the join
that already exists, the third is redundant and the redundancy is not
harmless:

    Q3 EPS, originally 1.20, restated to 1.05
      row 1: event_time=2024-09-30  available_time=2024-10-25  eps=1.20
      row 2: event_time=2024-09-30  available_time=2025-02-10  eps=1.05

    asof_join at 2024-12-01 -> 1.20      (what was known then)
    asof_join at 2025-03-01 -> 1.05      (the restatement)

A restatement is a ROW, not a column, and `available_time` on that row is
already its publication date. A separate `revision_time` would carry the
same value -- and, worse, it invites the encoding where one row per fact
holds a "last revised" stamp, which cannot reproduce history at all because
the earlier value is gone.

So the third thing worth declaring is not a timestamp. It is HOW REVISIONS
ARE ENCODED, because that is the difference between a source you can
backtest on and one you can only describe the present with.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from standard_quant_tools.error import ValidationError

#: The kinds of frame this contract describes. Prices are included even
#: though they are the one kind that is usually safe by construction: a bar
#: is knowable at its own close, so `event_time == available_time` and the
#: distinction collapses. Saying so explicitly is what stops somebody
#: assuming the same of a filing.
FRAME_KINDS = (
    "bars",
    "fundamentals",
    "estimates",
    "macro",
    "events",
    "corporate_actions",
    "reference_data",
)

#: How a source represents a value that was later restated.
#:
#: `versioned`  -- one row per version, each with its own `available_time`.
#:                 The only encoding that can reproduce a past decision.
#: `snapshot`   -- one row per fact, carrying the CURRENT value. History is
#:                 gone; a backtest on it silently uses numbers nobody had.
#: `none`       -- the source is never restated (a closing price, a split).
#: `unknown`    -- the provider does not say, which must be treated as
#:                 `snapshot` until somebody checks.
REVISION_ENCODINGS = ("versioned", "snapshot", "none", "unknown")


class TemporalContract(BaseModel):
    """
    What one source can tell you about when its facts became knowable.

    Deliberately a self-report rather than an inference. A provider that
    does not supply availability timestamps should say so and be refused,
    which is a better outcome than a heuristic that guesses right most of
    the time and produces a prescient backtest the rest.
    """

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="Provider or dataset name.")
    frame_kind: str = Field(..., description=f"One of {list(FRAME_KINDS)}.")
    has_event_time: bool = Field(
        ...,
        description="Whether the source says WHEN THE FACT IS ABOUT -- the "
        "quarter end, the reference month.",
    )
    has_available_time: bool = Field(
        ...,
        description="Whether the source says when the fact could first be "
        "ACTED ON. This is the join key, and the one that decides whether a "
        "point-in-time join is possible at all.",
    )
    entity_scoped: bool = Field(
        True,
        description="False for a global series -- CPI, Fed Funds, VIX -- "
        "which joins to every entity on each date rather than by entity.",
    )
    revisions: str = Field("unknown", description=f"One of {list(REVISION_ENCODINGS)}.")
    notes: List[str] = Field(default_factory=list)

    def model_post_init(self, _context) -> None:
        if self.frame_kind not in FRAME_KINDS:
            raise ValueError(
                f"unknown frame_kind {self.frame_kind!r}; expected one of "
                f"{list(FRAME_KINDS)}"
            )
        if self.revisions not in REVISION_ENCODINGS:
            raise ValueError(
                f"unknown revisions encoding {self.revisions!r}; expected one "
                f"of {list(REVISION_ENCODINGS)}"
            )

    @property
    def pit_safe(self) -> bool:
        """
        Whether a point-in-time join on this source is possible at all.

        Only `has_available_time` matters here. A source can lack
        `event_time` and still be joinable -- you simply cannot say what
        period the value describes, which is a labelling problem rather than
        a leak. Missing `available_time` is the leak.
        """
        return self.has_available_time

    @property
    def reproduces_history(self) -> bool:
        """
        Whether a past decision can be reproduced from this source.

        Stricter than `pit_safe`, and the two come apart in a way worth
        naming. A `snapshot` source with availability timestamps will join
        without leaking the FUTURE -- every row is stamped with when it was
        published -- but every restated value has been silently overwritten
        with its final version, so a backtest reads numbers that were never
        on anybody's screen. It is not lookahead; it is a different history.
        """
        return self.pit_safe and self.revisions in ("versioned", "none")

    def why_not_pit_safe(self) -> Optional[str]:
        """The specific reason, or None. Written for a caller to relay."""
        if self.pit_safe:
            return None
        return (
            f"{self.source!r} supplies no `available_time` for "
            f"{self.frame_kind!r}, so there is no way to know when each value "
            "became knowable. Joining on `event_time` instead would date a "
            "quarterly filing to the quarter end rather than its publication "
            "-- roughly three weeks of hindsight in every row, which reads as "
            "skill in the backtest."
        )

    def caveats(self) -> List[str]:
        """Everything worth telling a caller before they build on this."""
        out: List[str] = []
        reason = self.why_not_pit_safe()
        if reason:
            out.append(reason)
        if self.pit_safe and not self.reproduces_history:
            out.append(
                f"{self.source!r} stores {self.frame_kind!r} as "
                f"{self.revisions!r}: values that were later restated have "
                "been overwritten with their final version. The join will not "
                "leak the future, but a backtest will read numbers nobody had "
                "at the time. Treat results as indicative, not reproducible."
            )
        if self.pit_safe and not self.has_event_time:
            out.append(
                f"{self.source!r} supplies no `event_time` for "
                f"{self.frame_kind!r}, so the join is safe but the values "
                "cannot be attributed to the period they describe."
            )
        out.extend(self.notes)
        return out


def require_pit(contract: TemporalContract, purpose: str) -> None:
    """
    Refuse, by name, before any work is done.

    The point of failing here rather than at the join is cost. By the time
    `asof_join` refuses, a caller has already chosen a universe, fetched a
    history and cached it; the answer "this source could never have
    supported that" is the same answer, arriving after the expensive part.
    """
    reason = contract.why_not_pit_safe()
    if reason is None:
        return
    raise ValidationError(
        f"{purpose} needs point-in-time data. {reason} Either use a source "
        "that timestamps availability, or build this without it and say so."
    )


def contract_for_frame(
    frame, *, source: str, frame_kind: str, entity_scoped: bool = True
) -> TemporalContract:
    """
    Read the contract off a frame that is already in hand.

    Inspects columns rather than trusting a declaration, which is the right
    way round for data somebody has handed you: what a file claims about
    itself is a hope, and what its columns contain is a fact. `revisions` is
    inferred as `versioned` only when a duplicate (entity, event_time)
    genuinely carries more than one `available_time` -- seeing one version of
    everything is not evidence of anything, so it stays `unknown`.
    """
    from standard_quant_tools.modeling.dataset.point_in_time import (
        AVAILABLE_TIME,
        ENTITY,
        EVENT_TIME,
    )

    columns = set(getattr(frame, "columns", []))
    has_available = AVAILABLE_TIME in columns
    has_event = EVENT_TIME in columns

    revisions = "unknown"
    notes: List[str] = []
    if has_available and has_event:
        keys = [EVENT_TIME] + ([ENTITY] if entity_scoped and ENTITY in columns else [])
        try:
            versions = frame.groupby(keys)[AVAILABLE_TIME].nunique()
            if (versions > 1).any():
                revisions = "versioned"
                notes.append(
                    f"{int((versions > 1).sum())} fact(s) carry more than one "
                    "version, so restatements are preserved as separate rows "
                    "and a past decision can be reproduced."
                )
        except Exception:  # pragma: no cover - exotic index shapes
            pass

    return TemporalContract(
        source=source,
        frame_kind=frame_kind,
        has_event_time=has_event,
        has_available_time=has_available,
        entity_scoped=entity_scoped,
        revisions=revisions,
        notes=notes,
    )


def price_contract(source: str) -> TemporalContract:
    """
    The contract every bar series satisfies by construction.

    A daily bar is knowable at its own close, so `event_time` and
    `available_time` coincide and the distinction that matters everywhere
    else collapses here. This is stated rather than assumed because the
    assumption is exactly what gets carried over to filings, where it is
    false and expensive.

    ADJUSTMENT IS A SEPARATE QUESTION. A split-adjusted history is revised
    every time a split happens, and the adjusted close for 2019 is not the
    number anybody saw in 2019. That is `DataSetMetadata.adjusted`, and it
    is not what this contract describes.
    """
    return TemporalContract(
        source=source,
        frame_kind="bars",
        has_event_time=True,
        has_available_time=True,
        entity_scoped=True,
        revisions="none",
        notes=[
            "A bar is knowable at its own close, so event_time and "
            "available_time coincide. This is the one frame kind where that "
            "is true -- do not carry the assumption to filings.",
        ],
    )


__all__ = [
    "FRAME_KINDS",
    "REVISION_ENCODINGS",
    "TemporalContract",
    "contract_for_frame",
    "price_contract",
    "require_pit",
]
