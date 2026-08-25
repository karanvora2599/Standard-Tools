"""
Agentic Execution Analyst — what will this trade actually cost?

`get_liquidity_metrics` estimates the bid/ask spread from OHLCV bars via
Corwin-Schultz, and its own docstring says the result is a proxy — present
precisely because tick data is usually absent. A proxy cannot check itself.

This agent runs on the `portfolio` runtime, which holds the OHLCV proxies
and the tick measurements side by side on purpose: they are the same
question at two data fidelities, and `check_spread_proxy` is what compares
them. The DIRECTION of the proxy's error is what matters — overstating the
spread makes a backtest pessimistic, which is safe; understating it means
every backtest priced from it has been charging too little and reporting
returns that are too good.

REQUIRES A TICK FEED for the microstructure half. On a bar-only provider
those tools refuse by name rather than approximating from OHLCV, because a
"trade" derived from a bar is a fiction every measure downstream would
treat as fact. The agent is told to check first.

See Documentation/19_runtimes.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import _header, _log, run_agent, setup_logging

# ── Configuration ──────────────────────────────────────────────────
ANTHROPIC_API_KEY = ""  # Replace with your key
MODEL = "claude-haiku-4-5"

SYMBOL = "AAPL"
NOTIONAL = 2_000_000.0
TICK_WINDOW = ("2024-03-01 14:30:00", "2024-03-01 15:30:00")
BAR_WINDOW = ("2024-01-01", "2024-03-01")

SYSTEM_PROMPT = """You price execution: what a trade costs, and how much to trust that number.

FIRST, always: call describe_data_capabilities. The microstructure tools
need a provider with a tick feed and most environments do not have one. If
`trades` is false, say so plainly, use the OHLCV proxies
(get_liquidity_metrics) and state that they are proxies. Do NOT try to
approximate ticks from bars — spreads and signed order flow are not
recoverable from an OHLCV row, and a number invented that way would be
treated as a measurement by everything downstream.

With a tick feed:

1. get_microstructure_metrics. Distinguish the three spreads when you
   report them:
     - QUOTED is what crossing the book costs at an instant.
     - EFFECTIVE is what trades actually paid against the prevailing
       midpoint. This is the one a backtest should be charging.
     - The IMPACT and REALIZED halves imply opposite fixes: impact says
       trade smaller, realized says trade somewhere else.
   Prefer the SIZE-WEIGHTED averages when the question is about sizing a
   position, and say which you are quoting.
2. check_spread_proxy. Report the verdict and what it implies. If the proxy
   UNDERSTATES the measured spread, say directly that backtests priced from
   it have been charging too little and their returns are optimistic by
   roughly the ratio reported.
3. get_trade_profile. A book where most volume arrives in a few large
   prints behaves nothing like one where the same daily total arrives in
   thousands of small ones, at identical ADV — so an ADV participation
   limit means different things in the two.
4. estimate_trade_cost for the specific size asked about. Report
   breakeven_move_bps, which is the ROUND TRIP: an entry has to earn its
   exit's cost as well as its own.
5. get_capacity_report if the size looks large relative to volume.

Quotes are top of book only. No provider here exposes depth, so queue
position and resting size at a level are out of reach — say that rather
than estimating them."""

USER_REQUEST = f"""I want to trade ${NOTIONAL:,.0f} of {SYMBOL}.

What will it actually cost, how confident are you in that number, and is the
spread my backtests have been assuming anywhere close to what the tape says?

Tick window {TICK_WINDOW[0]} to {TICK_WINDOW[1]}; bars from {BAR_WINDOW[0]}
to {BAR_WINDOW[1]} for the proxy comparison."""


def main() -> None:
    log_file = setup_logging("agent_execution_analyst")

    _header("Agentic Execution Analyst")
    _log("Log file", str(log_file))
    _log("Symbol", SYMBOL)
    _log("Notional", f"${NOTIONAL:,.0f}")
    _log("Runtime", "portfolio (OHLCV proxies + tick measurements)")

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=ANTHROPIC_API_KEY,
        model=MODEL,
        max_iterations=20,
        # `portfolio` deliberately holds microstructure alongside
        # portfolio_risk: the proxy and the measurement answer the same
        # question, and check_spread_proxy needs both in scope.
        #
        # `meta` is joined because the FIRST instruction in the prompt is
        # describe_data_capabilities, which lives there. Without it the
        # agent would be told to check for a tick feed and then refused the
        # tool that checks — the runtime scope has to match the workflow
        # the prompt describes, or the prompt is asking for something the
        # agent cannot do.
        registry="portfolio+meta",
    )

    _header("EXECUTION ASSESSMENT")
    print(result)


if __name__ == "__main__":
    main()
