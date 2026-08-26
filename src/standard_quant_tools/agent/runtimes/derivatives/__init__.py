"""The `derivatives` runtime's registry: what it advertises and what it can
execute, built from one list so a tool cannot be advertised without being
dispatchable or the reverse.

WHY THIS IS ITS OWN RUNTIME. Option pricing lived in `research` under the
`analysis` category, which was right when it was two tools. It is twelve now,
and they answer a question `research` does not: `research` describes what an
asset IS, while these describe what a CONTRACT ON it is worth and what
holding it does to you. An agent scoped to one rarely wants the other.

The split clears the floor this library sets for one -- at least eight tools
in the new runtime, at least eight left in the donor. Derivatives holds
twelve; `research` was left with thirty-three and has grown since.

`get_option_pricing` and `get_implied_volatility` MOVED here from `research`,
which is a breaking change for anything scoped to that runtime. MOVED_FROM
carries the note so the failure says where they went rather than reading as
a hallucinated name.
"""

from standard_quant_tools.agent.models import (
    ImpliedVolatilityInput,
    OptionPricingInput,
)
from standard_quant_tools.agent.runtimes.research.tools import (
    get_implied_volatility,
    get_option_pricing,
)

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
from .tools import (
    analyze_option_strategy,
    analyze_vol_term_structure,
    check_put_call_parity,
    fit_volatility_smile,
    get_expected_move,
    get_implied_forward,
    get_option_greeks,
    get_option_risk_scenarios,
    get_volatility_cone,
    simulate_delta_hedge,
)

#: (name, description, input model) — the single source for both the
#: advertised schema and the dispatch table below.
TOOL_DEFS = [
    (
        "get_option_pricing",
        "Price a European option and return its first-order greeks, under "
        "Black-Scholes, Black-76, Bachelier or a binomial tree. `volatility` "
        "means different things to different models and no type system "
        "catches it: the lognormal models take a RELATIVE vol (a fraction of "
        "the underlying per year), Bachelier takes an ABSOLUTE one in the "
        "underlying's own units, and passing 0.30 to Bachelier on an $80 "
        "future means 30 cents of annual vol rather than 30%.",
        OptionPricingInput,
    ),
    (
        "get_implied_volatility",
        "The volatility that reproduces an observed option price. Solved by "
        "bisection on a monotone function, so it either converges or says it "
        "did not -- a price below intrinsic has no implied vol at all, and "
        "that is a refusal rather than a number.",
        ImpliedVolatilityInput,
    ),
    (
        "get_option_greeks",
        "The SECOND-ORDER greeks -- vanna, volga, charm and speed -- that "
        "explain why a delta-hedged book still loses money. Delta and gamma "
        "tell you today's risk; these tell you how that risk CHANGES. Vanna "
        "is why a vol spike forces a rehedge on a delta-flat book, volga is "
        "why short wings lose more than the vega number suggested, and charm "
        "is why a Friday delta-flat book opens Monday short with no move in "
        "the underlying. Units are stated per greek because there is no "
        "convention and the mismatch is a real source of error.",
        OptionGreeksInput,
    ),
    (
        "analyze_option_strategy",
        "The payoff, breakevens and aggregate greeks of an arbitrary "
        "multi-leg position -- any combination of calls, puts and stock, "
        "rather than a fixed menu of named structures. Breakevens are found "
        "numerically, and an unbounded loss is REPORTED as unbounded: a "
        "short call has no worst case, so returning the edge of the scanned "
        "range as 'max loss' would be a finite number standing in for an "
        "infinite risk.",
        OptionStrategyInput,
    ),
    (
        "fit_volatility_smile",
        "Fit a quoted smile as a quadratic in LOG-MONEYNESS and check it for "
        "arbitrage. Log-moneyness rather than strike because the smile is "
        "roughly symmetric in log(K/F) and emphatically not in K -- a "
        "parabola in strike puts its vertex at a fixed price, so the same "
        "shape refit after a 10% rally reports a different skew. Durrleman's "
        "condition is evaluated across the fitted range: a violation means "
        "the quotes admit a butterfly arbitrage, which in practice means one "
        "of them is stale.",
        VolatilitySmileInput,
    ),
    (
        "get_volatility_cone",
        "Where today's implied vol sits inside this name's own history of "
        "REALIZED vol, horizon by horizon. 'IV is 30' means nothing until you "
        "know this underlying's 30-day realized vol has spent two years "
        "between 18 and 55. Reports independent_windows next to each "
        "percentile, because rolling windows overlap and the observation "
        "count overstates the confidence by roughly the horizon.",
        VolatilityConeInput,
    ),
    (
        "analyze_vol_term_structure",
        "Contango or backwardation, and the FORWARD volatilities between "
        "expiries -- which is what a calendar spread actually prices. A "
        "trader seeing 30-day IV at 25 and 60-day at 28 is not being offered "
        "28 for the second month; they are being offered 30.6, whatever "
        "makes total variance add up. Trading off the quoted levels can "
        "reverse the sign of the position. Negative forward variance is "
        "reported as the calendar arbitrage it is.",
        VolTermStructureInput,
    ),
    (
        "check_put_call_parity",
        "Whether a call and a put on the same strike are mutually "
        "consistent, and what the violation is worth. This is a MODEL-FREE "
        "identity -- it follows from the payoffs alone and holds under any "
        "distribution -- which makes it the right first check on a quoted "
        "chain. The result names the likely causes before the arbitrage, "
        "because a break is a stale quote, a mismatched timestamp, a wrong "
        "dividend or a hard borrow far more often than it is free money; the "
        "implied dividend and forward are returned so the cause is "
        "identifiable rather than merely flagged.",
        PutCallParityInput,
    ),
    (
        "get_implied_forward",
        "The forward implied by carry, with financing, dividend and borrow "
        "broken out separately. When a quoted future disagrees with the "
        "computed forward the question is always WHICH term is wrong, and a "
        "single number cannot answer it. Borrow is kept apart from the "
        "dividend on purpose: a listed name's dividend is a known cash "
        "amount, while borrow floats and can move hundreds of basis points "
        "in a day on a squeezed stock.",
        ImpliedForwardInput,
    ),
    (
        "get_expected_move",
        "The move the option market is pricing over a horizon, with the "
        "standard misreading pre-empted. The number is ONE STANDARD "
        "DEVIATION and it gets quoted as 'the expected move' and then read "
        "as a bound -- under the model's own assumptions it is exceeded "
        "about 32% of the time, one earnings print in three. Both "
        "conventions are returned (the straddle approximation is 80% of the "
        "1-sd move) because confusing them misprices event trades. Pass "
        "realized_moves for the historical exceedance rate rather than the "
        "lognormal one.",
        ExpectedMoveInput,
    ),
    (
        "simulate_delta_hedge",
        "What a delta-hedged short option actually earns when the vol you "
        "sold at is not the vol that shows up. The expectation is known in "
        "closed form; the DISPERSION is not, and it is what decides whether "
        "the trade is sized correctly. Discrete hedging error scales as "
        "1/sqrt(n_hedges), so going from daily to twice-daily cuts the "
        "standard deviation by 29% rather than by half while doubling the "
        "transaction cost -- that tradeoff is the reason to simulate rather "
        "than compute.",
        DeltaHedgeInput,
    ),
    (
        "get_option_risk_scenarios",
        "A full REVALUATION grid over spot and volatility, not a delta-gamma "
        "approximation of one. Under a 20% move the Taylor estimate "
        "overstates a long call's gain by 5%, and by 11% at 30% -- the error "
        "grows with the cube of the move, which is why a stress test built "
        "on greeks understates a real gap. The two axes are shocked "
        "independently and the market does not move that way: read the "
        "down-spot/up-vol diagonal, not a row.",
        OptionScenariosInput,
    ),
]

TOOL_DISPATCH = {
    "get_option_pricing": (get_option_pricing, OptionPricingInput),
    "get_implied_volatility": (get_implied_volatility, ImpliedVolatilityInput),
    "get_option_greeks": (get_option_greeks, OptionGreeksInput),
    "analyze_option_strategy": (analyze_option_strategy, OptionStrategyInput),
    "fit_volatility_smile": (fit_volatility_smile, VolatilitySmileInput),
    "get_volatility_cone": (get_volatility_cone, VolatilityConeInput),
    "analyze_vol_term_structure": (
        analyze_vol_term_structure,
        VolTermStructureInput,
    ),
    "check_put_call_parity": (check_put_call_parity, PutCallParityInput),
    "get_implied_forward": (get_implied_forward, ImpliedForwardInput),
    "get_expected_move": (get_expected_move, ExpectedMoveInput),
    "simulate_delta_hedge": (simulate_delta_hedge, DeltaHedgeInput),
    "get_option_risk_scenarios": (
        get_option_risk_scenarios,
        OptionScenariosInput,
    ),
}

#: Every tool here belongs to the one category this runtime owns.
TOOL_CATEGORY = {name: "derivatives" for name in TOOL_DISPATCH}

__all__ = [
    "TOOL_CATEGORY",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "analyze_option_strategy",
    "analyze_vol_term_structure",
    "check_put_call_parity",
    "fit_volatility_smile",
    "get_expected_move",
    "get_implied_forward",
    "get_implied_volatility",
    "get_option_greeks",
    "get_option_pricing",
    "get_option_risk_scenarios",
    "get_volatility_cone",
    "simulate_delta_hedge",
]
