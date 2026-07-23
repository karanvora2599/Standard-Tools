"""
Dataset provenance metadata: what guarantees (or doesn't) the data behind
a backtest actually carries. A backtester's credibility depends as much on
data quality as strategy logic — this module makes explicit what a
provider does and doesn't promise, rather than leaving it implicit.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class DataSetMetadata(BaseModel):
    """
    Honest self-report from a data provider about one OHLCV pull. Every
    field reflects what the provider actually guarantees, not what would be
    ideal — e.g. YFinanceProvider reports survivorship_free=False and
    point_in_time=False because yfinance makes neither guarantee, not
    because the concepts don't apply.
    """
    provider: str = Field(..., description="Name of the data provider, e.g. 'yfinance'.")
    adjusted: bool = Field(..., description="Whether prices are split/dividend-adjusted.")
    survivorship_free: bool = Field(
        ..., description="Whether delisted/defunct securities remain queryable (no survivorship bias)."
    )
    point_in_time: bool = Field(
        ..., description="Whether historical values are guaranteed not to be silently revised after the fact."
    )
    frequency: str = Field(..., description="Bar interval, e.g. '1d'.")
    timezone: str = Field(..., description="Timezone the OHLCV timestamps are reported in.")
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp this metadata was generated.",
    )
