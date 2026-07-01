"""
Agentic backtest — Google Gemini 2.0 Flash.
The agent autonomously selects strategies, runs backtests, and summarizes findings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import setup_logging, run_agent, _header, _log

# ── Configuration ──────────────────────────────────────────────────
GEMINI_API_KEY = ""   # Replace with your key
MODEL          = "gemini-2.0-flash"

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
    log_file = setup_logging("agent_backtest_gemini")

    _header("Agentic Backtest — Gemini 2.0 Flash")
    _log("Log file", str(log_file))
    _log("Symbol",   "AAPL  |  2023")

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=GEMINI_API_KEY,
        model=MODEL,
        max_iterations=15,
    )

    _header("FINAL REPORT")
    print(result)
