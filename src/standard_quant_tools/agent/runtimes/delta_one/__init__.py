"""The `delta_one` runtime's registry: what it advertises and what it can
execute, built from one list so a tool cannot be advertised without being
dispatchable or the reverse.

WHY THIS IS ITS OWN RUNTIME. These questions were homeless. Carry lived in
`derivatives`, beta lived in `research`, sizing lived in `portfolio` and
execution lived in `microstructure`, and the question a Delta One desk
actually asks -- which of six instruments is the cheapest way to hold this
exposure -- needed all four and belonged to none. `derivatives` is the
nearest neighbour and is emphatically not the same runtime: it prices ONE
convex contract and asks what holding it does to you, while this prices the
RELATIONSHIP between several linear ones and asks which to hold.

NOTHING MOVED HERE. This runtime is new mathematics rather than a split, so
no donor was depleted and no MOVED_FROM entry was needed. In particular
`get_implied_forward` stays in `derivatives`, where put-call parity work
needs it -- this runtime calls the same LIBRARY function, which the
architecture permits because it scopes dispatch rather than values.

The floor this library sets for a runtime is eight tools. delta_one holds
nine, and the ninth is deliberate margin: shipping exactly at the floor
means one tool failing review makes the whole runtime unshippable.
"""

from .models import (
    BasisHistoryInput,
    CashFuturesBasisInput,
    CompareExpressionsInput,
    FuturesCurveInput,
    FuturesHedgeInput,
    HedgeEffectivenessInput,
    IndexBasketInput,
    RollAnalysisInput,
    SolveForwardCarryInput,
)
from .tools import (
    analyze_basis_history,
    analyze_cash_futures_basis,
    analyze_futures_curve,
    analyze_hedge_effectiveness,
    analyze_index_basket,
    analyze_roll,
    compare_delta_one_expressions,
    size_futures_hedge,
    solve_forward_carry,
)

#: (name, description, input model) -- the single source for both the
#: advertised schema and the dispatch table below.
#:
#: SENTENCE ONE IS THE WHOLE DESCRIPTION under `--tool-detail auto`, which
#: thins everything after it. Each one below is self-contained and names
#: the deliverable rather than opening with a preamble.
TOOL_DEFS = [
    (
        "analyze_cash_futures_basis",
        "A quoted future against its carry-fair value, with the mispricing "
        "attributed to financing, dividend or borrow. Returns the basis in "
        "POINTS, as an annualized rate, and as the financing the quote "
        "implies -- three views because points cannot be compared between a "
        "March and a December contract and the annualized spread can. A "
        "future 40 bps rich is usually expensive funding rather than edge, "
        "so the implied financing rate is the number to check against SOFR "
        "before trading it.",
        CashFuturesBasisInput,
    ),
    (
        "solve_forward_carry",
        "Recover whichever of financing, dividend or borrow a quoted forward "
        "implies, given the other two. One inverse rather than three tools, "
        "because they are one rearrangement of ln(F/S)/T = r - q - b and "
        "three near-identical names would be a coin flip for a model. The "
        "answer is CONDITIONAL and absorbs every error in the two rates you "
        "supply: an implied borrow computed against a wrong dividend is "
        "wrong by that whole dividend and looks entirely plausible.",
        SolveForwardCarryInput,
    ),
    (
        "analyze_basis_history",
        "Where today's basis sits inside its own history -- z-score, "
        "percentile, half-life and the distribution it came from. A basis of "
        "38 bps means nothing until you know the series has spent a year "
        "between -5 and +25, which is the difference between a number and a "
        "trade. Measured in bps of spot rather than points, because points "
        "are not comparable through time on anything that has moved. Without "
        "a window the z-score is full-sample and looks ahead.",
        BasisHistoryInput,
    ),
    (
        "analyze_futures_curve",
        "The futures term structure, and the FORWARD carry between expiries "
        "that a calendar spread actually prices. A trader seeing the near "
        "contract at 205 bps and the far at 210 is not being offered 210 for "
        "the period between them; they are offered whatever makes the two "
        "consistent, and trading off the quoted levels can reverse the sign "
        "of the position. Contango here describes the PRICE curve -- "
        "unrelated to the vol term structure that uses the same word.",
        FuturesCurveInput,
    ),
    (
        "analyze_roll",
        "What moving a position from one contract into the next actually "
        "costs, with the break-even rate it then has to out-earn. Distinct "
        "from the curve because this one has a size: the position is SIGNED, "
        "and a short rolled up a contango curve collects the step a long "
        "pays. Roll yield is reported as what it is -- a price step "
        "expressed as a rate, not a return, which a long gives up if spot "
        "does not move. Different multipliers are resized by money, not "
        "contract count.",
        RollAnalysisInput,
    ),
    (
        "size_futures_hedge",
        "The futures position that neutralizes a portfolio's beta, reported "
        "exact, rounded, and with the residual that rounding leaves. A "
        "-903.2 contract hedge is not available and -903 is; the 0.2 left "
        "over is $70,000 of unhedged beta and it decides whether the hedge "
        "is finished. Sizes on dollar beta, so a 1.12-beta book sells 12% "
        "more notional than it holds. Hedging with a different index needs "
        "that contract's own beta, or the hedge is short by the ratio.",
        FuturesHedgeInput,
    ),
    (
        "analyze_hedge_effectiveness",
        "Whether a hedge ratio actually removed risk, measured on realized "
        "returns rather than assumed from a beta. Reports volatility, beta "
        "and drawdown before and after, and the ROLLING ratio, which is the "
        "diagnostic that matters: a hedge whose ratio averaged 1.0 while "
        "ranging from 0.4 to 1.7 was never a hedge, it was two positions "
        "that averaged out, and its in-sample volatility reduction will not "
        "repeat. A hedge that raised volatility is almost always a sign "
        "error on the ratio.",
        HedgeEffectivenessInput,
    ),
    (
        "analyze_index_basket",
        "Value a basket of constituents against the index it replicates, "
        "attributing the spread name by name. A basket printing 40 bps from "
        "its index is far more often ONE constituent that has not traded "
        "than an arbitrage across all of them, so contributions come back "
        "sorted and a suspiciously unchanged price is flagged. Share-based "
        "with a divisor reproduces the index level; weight-based without one "
        "reproduces its returns but not its level, and conflating the two is "
        "how a basket comes out off by a constant.",
        IndexBasketInput,
    ),
    (
        "compare_delta_one_expressions",
        "Rank several ways of holding one exposure -- cash, ETF, future, "
        "forward, synthetic or swap -- on one annualized basis-point number. "
        "The HORIZON reorders them, and that is the answer rather than an "
        "artefact: execution is paid once and carry accrues, so a 2 bp round "
        "trip is 24 bp a year over one month and 1 bp over two years, and "
        "the cheapest instrument to hold is routinely not the cheapest to "
        "hold briefly. Every omitted cost term defaults to zero, so a "
        "surprisingly cheap row is usually an unpriced one.",
        CompareExpressionsInput,
    ),
]

TOOL_DISPATCH = {
    "analyze_cash_futures_basis": (
        analyze_cash_futures_basis,
        CashFuturesBasisInput,
    ),
    "solve_forward_carry": (solve_forward_carry, SolveForwardCarryInput),
    "analyze_basis_history": (analyze_basis_history, BasisHistoryInput),
    "analyze_futures_curve": (analyze_futures_curve, FuturesCurveInput),
    "analyze_roll": (analyze_roll, RollAnalysisInput),
    "size_futures_hedge": (size_futures_hedge, FuturesHedgeInput),
    "analyze_hedge_effectiveness": (
        analyze_hedge_effectiveness,
        HedgeEffectivenessInput,
    ),
    "analyze_index_basket": (analyze_index_basket, IndexBasketInput),
    "compare_delta_one_expressions": (
        compare_delta_one_expressions,
        CompareExpressionsInput,
    ),
}

#: Every tool here belongs to the one category this runtime owns.
TOOL_CATEGORY = {name: "delta_one" for name in TOOL_DISPATCH}

__all__ = [
    "TOOL_CATEGORY",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "analyze_basis_history",
    "analyze_cash_futures_basis",
    "analyze_futures_curve",
    "analyze_hedge_effectiveness",
    "analyze_index_basket",
    "analyze_roll",
    "compare_delta_one_expressions",
    "size_futures_hedge",
    "solve_forward_carry",
]
