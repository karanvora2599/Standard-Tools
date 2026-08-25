"""
Agentic backtest — Anthropic / Claude Haiku.
The agent autonomously selects strategies, runs backtests, and summarizes findings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import _header, _log, route_request, run_agent, setup_logging

# ── Configuration ──────────────────────────────────────────────────
ANTHROPIC_API_KEY = ""  # Replace with your key
MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """You are a quantitative analyst with access to a suite of backtesting and analysis tools.

Your goal is to:
1. Analyse the requested stock(s) using the available tools
2. Run multiple backtests to compare strategies
3. Evaluate risk metrics
4. Summarize your findings with a clear recommendation

Always start with technical analysis to understand the current market regime, then select and run
the most appropriate strategies. Compare results against buy-and-hold and provide a final recommendation.
"""

USER_REQUEST = (
    "Analyse AAPL for 2023. Run all four strategies (SMA, RSI, MACD, Bollinger), "
    "compare their performance against buy-and-hold, and tell me which works best and why."
)


if __name__ == "__main__":
    log_file = setup_logging("agent_backtest_anthropic")

    _header("Agentic Backtest — Claude Haiku")
    _log("Log file", str(log_file))
    _log("Symbol", "AAPL  |  2023")

    # Reference integration for the tool-category router (see
    # standard_quant_tools/agent/router.py): one cheap classification call
    # narrows the ~45-tool registry down to the categories this request
    # actually needs before the real agent loop starts, instead of handing
    # every tool to the model on every call. Fails open (routes to every
    # category) if the classification call errors or is unparseable.
    routed_categories = route_request(
        USER_REQUEST, api_key=ANTHROPIC_API_KEY, model=MODEL
    )

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=ANTHROPIC_API_KEY,
        model=MODEL,
        max_iterations=15,
        categories=routed_categories,
        # running a strategy is `backtest`; portfolio construction is not, and asking for it here is refused by name.
        # The router still narrows WITHIN the runtime; the runtime
        # is what makes the narrowing enforceable rather than
        # advisory. See Documentation/19_runtimes.md.
        registry="backtest",
    )

    _header("FINAL REPORT")
    print(result)
