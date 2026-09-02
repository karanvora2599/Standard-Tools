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

The floor this library sets for a runtime is eight tools. delta_one
shipped with nine -- the ninth deliberate margin, since shipping exactly
at the floor means one tool failing review makes the whole runtime
unshippable -- and now holds seventeen.

The six added second are the desk instruments: replication, ETF fair
value, total return swaps and futures, dividend points and index
rebalance flow. They were held back from the first nine because each
needs something the first nine did not -- a constrained optimizer, a
day-count convention, an index divisor -- and shipping the runtime on
the parts that needed none of that got it into use sooner.
"""

from .models import (
    BasisDislocationInput,
    BasisHistoryInput,
    BasisScanInput,
    CashFuturesBasisInput,
    CompareExpressionsInput,
    DividendPointsInput,
    EtfFairValueInput,
    FuturesCurveInput,
    FuturesHedgeInput,
    HedgeEffectivenessInput,
    IndexBasketInput,
    IndexRebalanceInput,
    ReplicationBasketInput,
    RollAnalysisInput,
    SolveForwardCarryInput,
    SpreadMonitorInput,
    TotalReturnFutureInput,
    TotalReturnSwapInput,
)
from .tools import (
    analyze_basis_history,
    analyze_cash_futures_basis,
    analyze_dividend_points,
    analyze_etf_fair_value,
    analyze_futures_curve,
    analyze_hedge_effectiveness,
    analyze_index_basket,
    analyze_index_rebalance,
    analyze_roll,
    analyze_total_return_future,
    compare_delta_one_expressions,
    detect_basis_dislocation,
    monitor_spread_stream,
    optimize_replication_basket,
    price_total_return_swap,
    scan_basis_dislocations,
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
    (
        "optimize_replication_basket",
        "The smallest basket that tracks a benchmark, minimizing the variance "
        "of the DIFFERENCE rather than the portfolio's own variance. Those are "
        "different portfolios: minimum-variance picks a defensive corner of a "
        "universe, minimum-tracking-error picks whatever most resembles the "
        "index. A max_names limit is enforced by thresholding because SLSQP "
        "cannot express an integer constraint and this library has no "
        "mixed-integer solver, so the answer is a GOOD basket of that size "
        "rather than a provably best one. Tracking error is in sample.",
        ReplicationBasketInput,
    ),
    (
        "analyze_etf_fair_value",
        "An ETF's premium or discount, and what survives the cost of capturing "
        "it. A visible 40 bp premium is almost never 40 bps of edge: the "
        "creation unit is an indivisible block, creating means crossing the "
        "BASKET's spreads rather than the fund's, and the NAV compared against "
        "is usually last night's struck value rather than a live one, so an "
        "intraday premium is mostly the market's move since the strike. The "
        "net figure after round-trip costs is the number that decides anything.",
        EtfFairValueInput,
    ),
    (
        "price_total_return_swap",
        "Mark a total return swap with the equity and financing legs "
        "separated. The payoff is simple -- receive price change plus "
        "dividends, pay a rate plus a spread -- and the CONVENTIONS are where "
        "the money is: ACT/360 accrues about 1.4% more financing than ACT/365F "
        "on the same period, and a zero dividend argument silently turns this "
        "into a price-return swap, understating the equity leg by 200 bps a "
        "year on a 2% yielder.",
        TotalReturnSwapInput,
    ),
    (
        "analyze_total_return_future",
        "Read a TRF quote as the financing spread it embeds, and compare it to "
        "what a swap charges. This is what answers 'regular futures imply 50 "
        "bp of funding and the TRF implies 95 -- where does the 45 go' as a "
        "calculation rather than a reconstruction. quote_convention is REQUIRED "
        "because some contracts quote the spread in bps and others quote a "
        "level, the two are not convertible without knowing which you have, "
        "and assuming wrong misprices by the entire financing leg.",
        TotalReturnFutureInput,
    ),
    (
        "analyze_dividend_points",
        "Index dividends as POINTS to a contract's expiry, dated and attributed "
        "by name. A continuous yield is the approximation this replaces: index "
        "dividends arrive in dense seasonal clusters, so a June and a September "
        "contract straddle the whole season and pricing both off one q puts one "
        "badly wrong. Supply a quoted future to get the market's own dividend "
        "number alongside the forecast -- a gap between them is usually a "
        "position rather than an error on either side.",
        DividendPointsInput,
    ),
    (
        "analyze_index_rebalance",
        "The buying and selling an index change forces on passive money, sized "
        "as DAYS OF VOLUME first and currency second. $2.8bn is nothing in a "
        "name trading $2bn a day and is a crisis in one trading $450m, and only "
        "the ratio says which. Index flow conventionally clears in a single "
        "closing auction that is 10-20% of the day, so leaving auction_fraction "
        "at 1.0 understates the binding participation by five to ten times. It "
        "sizes the flow and deliberately does not predict the price move.",
        IndexRebalanceInput,
    ),
    (
        "detect_basis_dislocation",
        "Whether a basis has STRUCTURALLY shifted rather than merely moved, "
        "by CUSUM and change-point detection. A z-score asks how "
        "unusual today is, and a basis that drifts two sigma wide and stays "
        "there never has a remarkable day -- CUSUM accumulates, so a "
        "sustained shift crosses when no single observation would. That is "
        "the difference between 'the basis is wide' and 'the basis is not "
        "the same basis any more', and only the second is a reason to "
        "re-examine the carry behind a position. A crossing is usually a roll "
        "or a dividend rather than a dislocation.",
        BasisDislocationInput,
    ),
    (
        "monitor_spread_stream",
        "Watch a spread on a LIVE feed, one stateful call at a time. A tool "
        "call returns, so there is nowhere for a subscription to live -- the "
        "state comes back in the result and goes in on the next call, which "
        "means a monitor can be paused, serialized and resumed elsewhere "
        "without losing its baseline. ONE tool covers live basis, ETF NAV, "
        "index arbitrage, roll spread and any cross-instrument spread, "
        "because those are three formulas rather than five. Accumulators "
        "are carried, so a hundred ticks in one call and a hundred calls of "
        "one tick agree exactly.",
        SpreadMonitorInput,
    ),
    (
        "scan_basis_dislocations",
        "Rank many spot/futures pairs by how far each basis sits from its "
        "OWN history, not by how wide it is. A name that always trades 40 "
        "bps is not news at 40 bps, so the ranking key is the z-score and "
        "the level is reported beside it. Optionally runs CUSUM per pair, "
        "because a basis sitting 2 sigma wide for months is a level while "
        "one that moved there last week is an event and they otherwise "
        "rank the same. Pairs it cannot evaluate are returned in `skipped` "
        "with a reason each rather than dropped, since a misaligned series "
        "is a data problem and not an absence of signal.",
        BasisScanInput,
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
    "optimize_replication_basket": (
        optimize_replication_basket,
        ReplicationBasketInput,
    ),
    "analyze_etf_fair_value": (analyze_etf_fair_value, EtfFairValueInput),
    "price_total_return_swap": (price_total_return_swap, TotalReturnSwapInput),
    "analyze_total_return_future": (
        analyze_total_return_future,
        TotalReturnFutureInput,
    ),
    "analyze_dividend_points": (analyze_dividend_points, DividendPointsInput),
    "analyze_index_rebalance": (analyze_index_rebalance, IndexRebalanceInput),
    "detect_basis_dislocation": (
        detect_basis_dislocation,
        BasisDislocationInput,
    ),
    "monitor_spread_stream": (monitor_spread_stream, SpreadMonitorInput),
    "scan_basis_dislocations": (scan_basis_dislocations, BasisScanInput),
}

#: Every tool here belongs to the one category this runtime owns.
TOOL_CATEGORY = {name: "delta_one" for name in TOOL_DISPATCH}

__all__ = [
    "TOOL_CATEGORY",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "analyze_basis_history",
    "analyze_cash_futures_basis",
    "analyze_dividend_points",
    "analyze_etf_fair_value",
    "analyze_futures_curve",
    "analyze_hedge_effectiveness",
    "analyze_index_basket",
    "analyze_index_rebalance",
    "analyze_roll",
    "analyze_total_return_future",
    "compare_delta_one_expressions",
    "detect_basis_dislocation",
    "monitor_spread_stream",
    "optimize_replication_basket",
    "price_total_return_swap",
    "size_futures_hedge",
    "solve_forward_carry",
]
