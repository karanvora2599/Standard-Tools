"""
Turning alpha scores into a portfolio, as a step you can stop at.

WHY THIS IS ITS OWN TOOL. The construction rules already exist in
`backtest/sizing.py` and are already used -- but only from inside larger
operations, where scores go in one end and a backtest comes out the other.
The question "given these scores, what portfolio do these rules produce"
had no answer that did not also run a simulation.

That matters most between modeling and backtesting. A model's predictions
become weights become a P&L, and if the P&L looks wrong there is no way to
tell whether the signal or the construction did it. Stopping in the middle
turns one opaque number into two inspectable ones.

    predictions -> construct_weights_from_scores -> LOOK AT THE WEIGHTS
                                                 -> only then simulate
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from standard_quant_tools.backtest import sizing
from standard_quant_tools.error import ValidationError

from ..handoff import publish, resolve

logger = logging.getLogger(__name__)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if math.isfinite(float(value)) else None
    return value


Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]

METHODS = ("rank", "top_bottom", "zscore", "vol_scaled")


class ConstructWeightsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores_ref: str = Field(
        ...,
        description=(
            "An `sqt://score_panel/...` or `sqt://predictions/...` "
            "reference. Bulk values cross runtimes as references, never "
            "through the conversation."
        ),
    )
    # A bare `str` here put the valid values only in the prose. The body
    # refused an unknown one, so nothing silently mis-ran -- but a model
    # reading the schema saw "string" and learned the four names only if it
    # read the description, and 'Rank' failed at call time rather than at
    # validation time. Literal puts them in the schema.
    method: Literal["rank", "top_bottom", "zscore", "vol_scaled"] = Field(
        "rank", description=f"One of: {', '.join(METHODS)}."
    )
    gross_leverage: float = Field(
        1.0, gt=0, description="Total absolute weight, summed across names."
    )
    n_long: Optional[int] = Field(
        None, gt=0, description="top_bottom only: how many names long."
    )
    n_short: Optional[int] = Field(
        None, ge=0, description="top_bottom only: how many names short."
    )
    returns_ref: Optional[str] = Field(
        None,
        description=(
            "vol_scaled only: an `sqt://returns_panel/...` to scale by. "
            "Required for that method and ignored by the others."
        ),
    )
    vol_lookback: int = Field(
        20, gt=1, description="vol_scaled only: the volatility window."
    )
    dollar_neutral: bool = Field(
        False,
        description=(
            "Shift each date's weights so longs and shorts cancel. Applied "
            "AFTER the method, so it changes net exposure and leaves the "
            "cross-sectional ordering alone."
        ),
    )
    run_id: str = Field(..., description="Groups this workflow's artifacts.")
    name: str = Field(..., description="Names the weight panel within the run.")


class WeightsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ref: str
    method: str
    n_dates: int = 0
    n_entities: int = 0
    gross_leverage: Stat = None
    net_exposure: Stat = Field(
        None, description="Longs minus shorts on the last date. 0 if neutral."
    )
    max_weight: Stat = None
    min_weight: Stat = None
    n_long: int = 0
    n_short: int = 0
    latest: Dict[str, Stat] = Field(
        default_factory=dict, description="The final date's weights, by name."
    )
    warnings: List[str] = Field(default_factory=list)


def _score_frame(ref: str) -> pd.DataFrame:
    try:
        data = resolve(ref)
    except Exception as exc:  # noqa: BLE001 -- one refusal, not a traceback
        raise ValidationError(
            f"{ref!r} could not be resolved as a score panel: {exc}"
        ) from exc
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValidationError(
            f"{ref!r} did not resolve to a non-empty date-by-entity frame."
        )
    return data


def construct_weights_from_scores(
    input_data: ConstructWeightsInput,
) -> WeightsResult:
    """Alpha scores into portfolio weights, inspectable before any simulation."""
    if input_data.method not in METHODS:
        raise ValidationError(
            f"unknown method {input_data.method!r}; expected one of "
            f"{list(METHODS)}."
        )
    scores = _score_frame(input_data.scores_ref)

    if input_data.method == "rank":
        weights = sizing.rank_weighted(scores, input_data.gross_leverage)
    elif input_data.method == "zscore":
        weights = sizing.zscore_normalized(scores, input_data.gross_leverage)
    elif input_data.method == "top_bottom":
        if input_data.n_long is None:
            raise ValidationError(
                "method='top_bottom' needs n_long -- how many names to hold "
                "long is the whole parameter, and there is no sane default."
            )
        weights = sizing.equal_weight_top_bottom(
            scores,
            input_data.n_long,
            input_data.n_short if input_data.n_short is not None else 0,
            input_data.gross_leverage,
        )
    else:  # vol_scaled
        if not input_data.returns_ref:
            raise ValidationError(
                "method='vol_scaled' needs returns_ref: the scaling divides "
                "by each name's realized volatility, which cannot be "
                "recovered from the scores."
            )
        returns = _score_frame(input_data.returns_ref)
        weights = sizing.vol_scaled(
            scores, returns, input_data.vol_lookback, input_data.gross_leverage
        )

    warnings: List[str] = []
    if input_data.dollar_neutral:
        weights = sizing.dollar_neutral(weights)
        warnings.append(
            "Dollar-neutralised AFTER construction, so net exposure is ~0 "
            "and gross may differ from the requested leverage -- the shift "
            "preserves ordering, not scale."
        )

    ref = publish(
        weights,
        kind="weight_panel",
        run_id=input_data.run_id,
        name=input_data.name,
        producer="construct_weights_from_scores",
    )

    last = weights.iloc[-1].dropna()
    return WeightsResult(
        ref=ref,
        method=input_data.method,
        n_dates=int(len(weights)),
        n_entities=int(weights.shape[1]),
        gross_leverage=float(last.abs().sum()),
        net_exposure=float(last.sum()),
        max_weight=float(last.max()) if len(last) else None,
        min_weight=float(last.min()) if len(last) else None,
        n_long=int((last > 0).sum()),
        n_short=int((last < 0).sum()),
        latest={str(k): float(v) for k, v in last.items()},
        warnings=warnings
        + [
            "These are TARGET weights, not a P&L. What they earn depends on "
            "costs, fills and rebalancing, which run_portfolio_simulation "
            "applies and this tool deliberately does not."
        ],
    )


__all__ = [
    "METHODS",
    "ConstructWeightsInput",
    "WeightsResult",
    "construct_weights_from_scores",
]
