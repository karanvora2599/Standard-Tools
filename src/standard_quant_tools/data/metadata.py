"""
Dataset provenance metadata: what guarantees (or doesn't) the data behind
a backtest actually carries. A backtester's credibility depends as much on
data quality as strategy logic — this module makes explicit what a
provider does and doesn't promise, rather than leaving it implicit.
"""

from datetime import datetime, timezone
from typing import Any, List, Optional

from pydantic import BaseModel, Field, computed_field, model_validator

#: The three answers to "do returns computed from these bars include
#: dividends". Read off `dividend_adjusted` alone: whether splits are
#: taken out is a different question, which the split screen asks.
TOTAL_RETURN = "total_return"
PRICE_RETURN = "price_return"
UNKNOWN_RETURN_BASIS = "unknown"
RETURN_BASES = (TOTAL_RETURN, PRICE_RETURN, UNKNOWN_RETURN_BASIS)


def all_of(*flags: Optional[bool]) -> Optional[bool]:
    """Three-valued AND: False when any flag is False, True when every flag
    is True, otherwise None (not known)."""
    if any(flag is False for flag in flags):
        return False
    if flags and all(flag is True for flag in flags):
        return True
    return None


def return_basis_for(dividend_adjusted: Optional[bool]) -> str:
    """`total_return` for dividend-adjusted bars, `price_return` for bars
    that leave dividends in the price, `unknown` when nobody said."""
    if dividend_adjusted is True:
        return TOTAL_RETURN
    if dividend_adjusted is False:
        return PRICE_RETURN
    return UNKNOWN_RETURN_BASIS


class DataSetMetadata(BaseModel):
    """
    Honest self-report from a data provider about one OHLCV pull. Every
    field reflects what the provider actually guarantees, not what would be
    ideal — e.g. YFinanceProvider reports survivorship_free=False and
    point_in_time=False because yfinance makes neither guarantee, not
    because the concepts don't apply.

    ADJUSTMENT IS TWO QUESTIONS. One `adjusted` flag used to answer both,
    and the providers that said True meant different things by it: yfinance
    requests total-return bars (split- and dividend-adjusted), Polygon's
    `adjusted=true` takes out splits only, and Bloomberg's dividends
    followed whatever the terminal's own settings said. A return computed
    from one is not the return computed from another, by the dividend
    yield. So `split_adjusted` and `dividend_adjusted` are stated
    separately, each True, False or None (not known, with the reason in
    `adjustment_note`), and `adjusted` is their AND, kept for callers that
    read it. `return_basis` names what a return computed from the bars is.

    A caller that constructs this with `adjusted` alone is read at that
    field's documented word -- "split/dividend-adjusted" -- so True sets
    both flags and False sets neither; the library's own providers state
    the two flags themselves.
    """

    provider: str = Field(
        ..., description="Name of the data provider, e.g. 'yfinance'."
    )
    split_adjusted: Optional[bool] = Field(
        None,
        description=(
            "Whether prices are adjusted for splits (a 10:1 split is not a "
            "-90% bar). None when the provider cannot say."
        ),
    )
    dividend_adjusted: Optional[bool] = Field(
        None,
        description=(
            "Whether prices are adjusted for cash dividends, so that a "
            "return computed from them is a total return. None when the "
            "provider cannot say."
        ),
    )
    adjusted: Optional[bool] = Field(
        None,
        description=(
            "split_adjusted AND dividend_adjusted: True only when both are, "
            "False when either is not, None when neither answer rules it. "
            "Kept for callers that read one flag; the two above are the "
            "statement."
        ),
    )
    adjustment_note: Optional[str] = Field(
        None,
        description=(
            "What the two flags rest on: the request parameter sent and "
            "the vendor's documented meaning of it, and why a flag is None "
            "when it is."
        ),
    )
    survivorship_free: bool = Field(
        ...,
        description="Whether delisted/defunct securities remain queryable (no survivorship bias).",
    )
    point_in_time: bool = Field(
        ...,
        description="Whether historical values are guaranteed not to be silently revised after the fact.",
    )
    frequency: str = Field(..., description="Bar interval, e.g. '1d'.")
    timezone: str = Field(
        ...,
        description=(
            "The zone the bars are STATED in. Every provider's index is "
            "normalised before it is returned: daily bars are naive session "
            "dates in this zone, intraday bars are naive UTC instants. "
            "Localising the index to this zone before a join yields nothing "
            "aligned; it is a label, not an instruction."
        ),
    )
    notes: List[str] = Field(
        default_factory=list,
        description="What the flags cannot say: which feed answers, what the "
        "index is, what the provider refuses.",
    )
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp this metadata was generated.",
    )

    @model_validator(mode="before")
    @classmethod
    def _two_flags_and_their_and(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        stated = "split_adjusted" in data or "dividend_adjusted" in data
        if stated:
            derived = all_of(data.get("split_adjusted"), data.get("dividend_adjusted"))
            given = data.get("adjusted")
            if given is not None and given != derived:
                raise ValueError(
                    f"adjusted={given!r} contradicts split_adjusted="
                    f"{data.get('split_adjusted')!r} and dividend_adjusted="
                    f"{data.get('dividend_adjusted')!r}, whose AND is "
                    f"{derived!r}. Pass the two flags and leave adjusted out; "
                    "it is derived from them."
                )
            return {**data, "adjusted": derived}
        if "adjusted" in data:
            legacy = data.get("adjusted")
            return {**data, "split_adjusted": legacy, "dividend_adjusted": legacy}
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def return_basis(self) -> str:
        """What a return computed from these bars is: `total_return`
        (dividends in), `price_return` (dividends out) or `unknown`."""
        return return_basis_for(self.dividend_adjusted)
