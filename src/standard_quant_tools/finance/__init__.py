"""
Finance on a set of holdings: positions valued in one base currency.

`position_values` values each position (local notional = quantity x price x
multiplier, base notional = local x the caller's FX rate, and its side);
`exposure_summary` sums them into gross, net, long and short in the base
currency and the exposure per currency. Both are pure functions of their
arguments: no rate or price is fetched. `currency_code` reads an ISO 4217
code, refusing an unknown one with the nearest valid code.
"""

from .currency import (
    ACTIVE_CURRENCIES,
    WITHDRAWN_CURRENCIES,
    currency_code,
    nearest_currency,
)
from .positions import (
    MAX_POSITIONS,
    CurrencyExposure,
    ExposureSummary,
    PositionValue,
    exact_product,
    exposure_summary,
    position_values,
)

__all__ = [
    "ACTIVE_CURRENCIES",
    "MAX_POSITIONS",
    "WITHDRAWN_CURRENCIES",
    "CurrencyExposure",
    "ExposureSummary",
    "PositionValue",
    "currency_code",
    "exact_product",
    "exposure_summary",
    "nearest_currency",
    "position_values",
]
