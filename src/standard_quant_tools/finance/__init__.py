"""
Finance on a set of holdings: positions valued in one base currency.

`position_values` values each position (local notional = quantity x price x
multiplier, base notional = local x the caller's FX rate, and its side);
`exposure_summary` sums them into gross, net, long and short in the base
currency and the exposure per currency. Both are pure functions of their
arguments: no rate or price is fetched. `currency_code` reads an ISO 4217
code, refusing an unknown one with the nearest valid code.

`financing_cost` prices holdings' financing over a horizon under a
`backtest.financing.FinancingSpec`; `margin_requirement` gives Reg T
initial and FINRA 4210 maintenance margin per line; `margin_runway` walks
a stress path to the first maintenance call and the buffer that avoids it.

`currency_exposure` adds FX hedges to that exposure, with each currency's
hedge ratio and the local / FX / cross split of each position's return;
`book_carry` is the carry of each line and of the set; `fx_forward_hedge`
prices the covered-interest-parity forward that hedges a foreign currency
and the hedged return it gives.
"""

from .carry import book_carry, carry_kind
from .cash_flows import (
    CashFlowSchedule,
    RateRoots,
    analyze_cash_flows,
    dated_cash_flows,
    periodic_cash_flows,
    rate_roots,
)
from .cost_of_capital import estimate_cost_of_capital, relever_beta, unlever_beta
from .currency import (
    ACTIVE_CURRENCIES,
    WITHDRAWN_CURRENCIES,
    currency_code,
    nearest_currency,
)
from .currency_exposure import currency_exposure, hedge_legs
from .financing_cost import FinancingCost, ShortBorrow, financing_cost
from .fx_forward import (
    covered_interest_forward,
    currency_pair,
    fx_forward_hedge,
    leg_year_fraction,
    pip_size_for,
)
from .margin import (
    MarginCall,
    MarginLine,
    MarginRequirement,
    MarginRunway,
    margin_requirement,
    margin_runway,
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
from .valuation import dcf_valuation

__all__ = [
    "ACTIVE_CURRENCIES",
    "MAX_POSITIONS",
    "WITHDRAWN_CURRENCIES",
    "CashFlowSchedule",
    "CurrencyExposure",
    "ExposureSummary",
    "FinancingCost",
    "MarginCall",
    "MarginLine",
    "MarginRequirement",
    "MarginRunway",
    "PositionValue",
    "RateRoots",
    "analyze_cash_flows",
    "ShortBorrow",
    "book_carry",
    "carry_kind",
    "covered_interest_forward",
    "currency_code",
    "dated_cash_flows",
    "dcf_valuation",
    "estimate_cost_of_capital",
    "currency_exposure",
    "currency_pair",
    "exact_product",
    "exposure_summary",
    "financing_cost",
    "margin_requirement",
    "margin_runway",
    "fx_forward_hedge",
    "hedge_legs",
    "leg_year_fraction",
    "nearest_currency",
    "periodic_cash_flows",
    "pip_size_for",
    "position_values",
    "rate_roots",
    "relever_beta",
    "unlever_beta",
]
