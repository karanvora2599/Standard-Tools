"""
The contract every registered label satisfies.

WHAT A TARGET IS, AS A RECORD. `TARGET_KINDS` already said what each label
was -- which tasks consume it, whether prices can build it, whether it is
continuous -- and said so once. What it could not say was HOW to build it:
that lived in a chain of `if spec.type == ...` branches in
`dataset/target.py`, plus a frozenset naming the two labels that need the
whole cross-section, plus a hand-written Literal pinned equal to the dict
by test. Adding a label meant editing all four, and a firm's own label --
a residual return, an earnings drift, an execution shortfall -- could not
be added at all without editing the library.

So a definition carries the how beside the what:

    builder(ohlcv, spec, context)           -> Series on the entity's bars
    label_end_builder(ohlcv, spec, context) -> when each row's label closes
    cross_sectional_stage(panel, spec, col) -> the panel-wide step, if any

`builder` sees the entity's FULL OHLCV and the same `FeatureContext` a
feature sees (benchmark close, interval), so a label that needs High and
Low, or the benchmark's own return, can be written without reaching past
the contract. `label_end_builder` defaults to "horizon bars ahead on the
entity's own calendar", which is what the engine's purge reads; a label
that can close early -- a barrier, a fill -- supplies its own so the purge
removes exactly the rows that overlap and not the ones a nominal horizon
would. `cross_sectional_stage` is for a label defined against the OTHER
entities on the date, which cannot exist until every entity is stacked.

WHAT AN EXTERNAL LABEL IS NOT. A markout, a fill probability, a time to
fill: recorded, never computed. A definition with `buildable=False` must
have no builder, and the registry refuses one that does -- a bar-derived
approximation of a fill probability would be a number with nothing behind
it, and it would look exactly like a number with something behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from standard_quant_tools.modeling.estimators.bounds import EstimatorParamSchema


@dataclass(frozen=True)
class TargetKind:
    """The what of a label, as every consumer before the registry read it.
    A view onto a definition, kept so those consumers need no change."""

    #: The tasks that can be fitted against it. A continuous label suits a
    #: regressor and a ranker; a discrete one suits a classifier.
    tasks: Tuple[str, ...]
    #: Whether `build_target` can produce it from an OHLCV frame. FALSE for
    #: every microstructure and execution label.
    buildable: bool
    #: Continuous labels reject a `threshold`, which only means something
    #: for a binarized one.
    continuous: bool
    description: str


class TargetDefinition(BaseModel):
    """One registry entry: the what, the how, and the bounds on its
    parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    description: str
    tasks: Tuple[str, ...]
    buildable: bool
    continuous: bool
    #: OHLCV columns the builder reads. Checked against the fetched frame
    #: before anything is built, the way a feature's `requires` is.
    requires: List[str] = Field(default_factory=lambda: ["Close"])
    builder: Optional[Callable[..., Any]] = None
    label_end_builder: Optional[Callable[..., Any]] = None
    cross_sectional_stage: Optional[Callable[..., Any]] = None
    #: Bounds for `TargetSpec.params`, the way an estimator's schema bounds
    #: its params. Empty for the built-ins, whose parameters are spec
    #: fields; a custom label declares its own.
    param_schema: EstimatorParamSchema = Field(default_factory=EstimatorParamSchema)
    default_params: Dict[str, Any] = Field(default_factory=dict)

    @property
    def kind(self) -> TargetKind:
        return TargetKind(
            tasks=tuple(self.tasks),
            buildable=self.buildable,
            continuous=self.continuous,
            description=self.description,
        )

    @property
    def cross_sectional(self) -> bool:
        return self.cross_sectional_stage is not None


__all__ = ["TargetDefinition", "TargetKind"]
