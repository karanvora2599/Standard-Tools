"""
The `derivatives` runtime: what an option is worth and what holding it does
to you.

Pricing, the greeks that explain why a hedged book still loses money, the
consistency of a quoted surface, and what a hedge costs to run. Nothing here
fetches an option chain -- the library has no options data provider, and a
tool that pretended to would compute a chain that does not exist. Quotes
come in as arguments, which also means the same tools work on a hypothetical
surface, which is most of what they are used for.

THE FUNCTIONS ARE THIN, and deliberately so. Every number comes from
`analysis/derivatives.py`; this module converts JSON-shaped arguments into
what those expect and wraps the answer in a typed result. It computes
nothing, because a second implementation is a second thing to keep correct.

RESULTS ARE TYPED rather than passed through as dicts. The MCP server builds
its structured-output schema from the return annotation, so an untyped return
means a client receives JSON it has no schema for -- and on this surface the
schema is carrying real information, like the fact that `vega` is per
volatility POINT.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from standard_quant_tools.analysis import derivatives as lib
from standard_quant_tools.error import ValidationError

from .models import (
    DeltaHedgeInput,
    ExpectedMoveInput,
    ImpliedForwardInput,
    OptionGreeksInput,
    OptionScenariosInput,
    OptionStrategyInput,
    PutCallParityInput,
    VolatilityConeInput,
    VolatilitySmileInput,
    VolTermStructureInput,
)
from .results import (
    DeltaHedgeResult,
    ExpectedMoveResult,
    ImpliedForwardResult,
    OptionGreeksResult,
    OptionScenariosResult,
    OptionStrategyResult,
    PutCallParityResult,
    VolatilityConeResult,
    VolatilitySmileResult,
    VolTermStructureResult,
)

logger = logging.getLogger(__name__)


def _numeric_keys(mapping: Dict[str, float], field: str) -> Dict[float, float]:
    """
    JSON object keys are strings; these maps are keyed by a NUMBER.

    Converted here with the failing key named, because a silent skip would
    drop an expiry from a term structure and produce a shorter, plausible,
    wrong answer.
    """
    out: Dict[float, float] = {}
    for key, value in mapping.items():
        try:
            out[float(key)] = float(value)
        except (TypeError, ValueError):
            raise ValidationError(
                f"{field}: key {key!r} is not a number. This map is keyed by "
                "a numeric value written as a string (JSON has no numeric "
                "keys), e.g. {'0.0833': 0.24}."
            ) from None
    return out


def get_option_greeks(input_data: OptionGreeksInput) -> OptionGreeksResult:
    return OptionGreeksResult(
        **lib.option_greeks(
            spot=input_data.spot,
            strike=input_data.strike,
            time_to_expiry=input_data.time_to_expiry,
            volatility=input_data.volatility,
            risk_free_rate=input_data.risk_free_rate,
            option_type=input_data.option_type,
            dividend_yield=input_data.dividend_yield,
        )
    )


def analyze_option_strategy(input_data: OptionStrategyInput) -> OptionStrategyResult:
    legs = [leg.model_dump(exclude_none=True) for leg in input_data.legs]
    return OptionStrategyResult(
        **lib.analyze_strategy(
            legs,
            spot=input_data.spot,
            risk_free_rate=input_data.risk_free_rate,
            dividend_yield=input_data.dividend_yield,
            spot_range=input_data.spot_range,
        )
    )


def fit_volatility_smile(input_data: VolatilitySmileInput) -> VolatilitySmileResult:
    return VolatilitySmileResult(
        **lib.fit_volatility_smile(
            input_data.strikes,
            input_data.implied_vols,
            forward=input_data.forward,
            time_to_expiry=input_data.time_to_expiry,
        )
    )


def get_volatility_cone(input_data: VolatilityConeInput) -> VolatilityConeResult:
    implied = None
    if input_data.current_implied:
        implied = {
            int(key): value
            for key, value in _numeric_keys(
                input_data.current_implied, "current_implied"
            ).items()
        }
    return VolatilityConeResult(
        **lib.volatility_cone(
            pd.Series(input_data.prices),
            horizons=input_data.horizons or lib.CONE_HORIZONS,
            current_implied=implied,
        )
    )


def analyze_vol_term_structure(
    input_data: VolTermStructureInput,
) -> VolTermStructureResult:
    return VolTermStructureResult(
        **lib.analyze_vol_term_structure(
            _numeric_keys(input_data.implied_by_expiry, "implied_by_expiry")
        )
    )


def check_put_call_parity(input_data: PutCallParityInput) -> PutCallParityResult:
    return PutCallParityResult(
        **lib.check_put_call_parity(
            call_price=input_data.call_price,
            put_price=input_data.put_price,
            spot=input_data.spot,
            strike=input_data.strike,
            time_to_expiry=input_data.time_to_expiry,
            risk_free_rate=input_data.risk_free_rate,
            dividend_yield=input_data.dividend_yield,
            tolerance_bps=input_data.tolerance_bps,
        )
    )


def get_implied_forward(input_data: ImpliedForwardInput) -> ImpliedForwardResult:
    return ImpliedForwardResult(
        **lib.implied_forward_price(
            spot=input_data.spot,
            time_to_expiry=input_data.time_to_expiry,
            risk_free_rate=input_data.risk_free_rate,
            dividend_yield=input_data.dividend_yield,
            borrow_rate=input_data.borrow_rate,
        )
    )


def get_expected_move(input_data: ExpectedMoveInput) -> ExpectedMoveResult:
    return ExpectedMoveResult(
        **lib.expected_move(
            spot=input_data.spot,
            implied_vol=input_data.implied_vol,
            days=input_data.days,
            realized_moves=input_data.realized_moves,
        )
    )


def simulate_delta_hedge(input_data: DeltaHedgeInput) -> DeltaHedgeResult:
    return DeltaHedgeResult(
        **lib.simulate_delta_hedge(
            spot=input_data.spot,
            strike=input_data.strike,
            time_to_expiry=input_data.time_to_expiry,
            implied_vol=input_data.implied_vol,
            realized_vol=input_data.realized_vol,
            risk_free_rate=input_data.risk_free_rate,
            option_type=input_data.option_type,
            n_hedges=input_data.n_hedges,
            n_paths=input_data.n_paths,
            transaction_cost_bps=input_data.transaction_cost_bps,
            seed=input_data.seed,
        )
    )


def get_option_risk_scenarios(
    input_data: OptionScenariosInput,
) -> OptionScenariosResult:
    kwargs: Dict[str, Any] = {
        "spot": input_data.spot,
        "strike": input_data.strike,
        "time_to_expiry": input_data.time_to_expiry,
        "volatility": input_data.volatility,
        "risk_free_rate": input_data.risk_free_rate,
        "option_type": input_data.option_type,
        "quantity": input_data.quantity,
        "days_forward": input_data.days_forward,
    }
    # Passed only when supplied, so the library's own defaults stay the
    # single place those grids are defined.
    if input_data.spot_shocks is not None:
        kwargs["spot_shocks"] = input_data.spot_shocks
    if input_data.vol_shocks is not None:
        kwargs["vol_shocks"] = input_data.vol_shocks
    return OptionScenariosResult(**lib.option_risk_scenarios(**kwargs))


__all__ = [
    "analyze_option_strategy",
    "analyze_vol_term_structure",
    "check_put_call_parity",
    "fit_volatility_smile",
    "get_expected_move",
    "get_implied_forward",
    "get_option_greeks",
    "get_option_risk_scenarios",
    "get_volatility_cone",
    "simulate_delta_hedge",
]
