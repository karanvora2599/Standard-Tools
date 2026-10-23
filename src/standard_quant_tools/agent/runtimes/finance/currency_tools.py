"""
The currency tools of the finance runtime: how much of a multi-currency set
of positions is exposed to each currency once its FX hedges are counted,
what the set carries, and the FX forward that hedges a foreign currency.

- `get_currency_exposure` -- exposure per currency in the base currency,
  net and gross, after forward or spot FX hedges given as pair notionals;
  each currency's hedge ratio; and, with each position's local return and
  each currency's FX return, the split of every position's base-currency
  return into local, FX and cross terms that add up exactly.
- `analyze_book_carry` -- the carry of each line and of the set (Koijen,
  Moskowitz, Pedersen & Vrugt 2018): (S - F) / (F T) for a future or
  forward, income less financing (and borrow when short) for a funded
  holding, and the notional-weighted carry of the set.
- `price_fx_forward_hedge` -- the covered-interest-parity forward with each
  leg on its own day count and any cross-currency basis on the foreign leg,
  its points in pips, the annualized pickup or cost of the hedge, and a
  foreign asset's hedged against unhedged return.

THEY TAKE POSITIONS OR A PAIR, NEVER TICKERS: every price, FX rate, rate and
day count is the caller's and nothing is fetched, so the same input gives
the same bits. The library functions are
`finance.currency_exposure.currency_exposure`, `finance.carry.book_carry`
and `finance.fx_forward.fx_forward_hedge`; the models are in
`currency_models`.
"""

from __future__ import annotations

from standard_quant_tools.finance.carry import book_carry
from standard_quant_tools.finance.currency_exposure import currency_exposure
from standard_quant_tools.finance.fx_forward import fx_forward_hedge

from .currency_models import (
    BookCarryInput,
    BookCarryResult,
    CurrencyExposureInput,
    CurrencyExposureResult,
    FxForwardHedgeInput,
    FxForwardHedgeResult,
)


def get_currency_exposure(input_data: CurrencyExposureInput) -> CurrencyExposureResult:
    return CurrencyExposureResult(
        **currency_exposure(
            input_data.positions,
            input_data.base_currency,
            input_data.fx_rates,
            hedges=input_data.hedges,
            fx_returns=input_data.fx_returns,
        )
    )


def analyze_book_carry(input_data: BookCarryInput) -> BookCarryResult:
    return BookCarryResult(
        **book_carry(
            input_data.positions, input_data.base_currency, input_data.fx_rates
        )
    )


def price_fx_forward_hedge(input_data: FxForwardHedgeInput) -> FxForwardHedgeResult:
    return FxForwardHedgeResult(
        **fx_forward_hedge(**input_data.model_dump(exclude_none=True))
    )


TOOL_DEFS = [
    (
        "get_currency_exposure",
        "Currency exposure of a set of positions in the base currency, net and "
        "gross per currency after FX forward or spot hedges, with each "
        "currency's hedge ratio. With each position's local_return and each "
        "currency's fx_return, splits every base-currency return into local, "
        "FX and cross terms that add up exactly.",
        CurrencyExposureInput,
    ),
    (
        "analyze_book_carry",
        "Annual carry of each position and of the set in the base currency "
        "(Koijen, Moskowitz, Pedersen & Vrugt 2018). A line with spot and "
        "years_to_expiry is a future or forward: (S - F)/(F T). One with "
        "income_rate and financing_rate is funded: income less financing, "
        "less borrow when short. A short earns the opposite of a long.",
        BookCarryInput,
    ),
    (
        "price_fx_forward_hedge",
        "Covered-interest-parity FX forward of a pair, each leg on its own day "
        "count plus any cross-currency basis, its points in pips, and the "
        "annual pickup or cost of the hedge. The basis is added to the "
        "foreign leg. With an asset return and the spot at delivery, the "
        "hedged against unhedged return.",
        FxForwardHedgeInput,
    ),
]

TOOL_DISPATCH = {
    "get_currency_exposure": (get_currency_exposure, CurrencyExposureInput),
    "analyze_book_carry": (analyze_book_carry, BookCarryInput),
    "price_fx_forward_hedge": (price_fx_forward_hedge, FxForwardHedgeInput),
}

__all__ = [
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "analyze_book_carry",
    "get_currency_exposure",
    "price_fx_forward_hedge",
]
