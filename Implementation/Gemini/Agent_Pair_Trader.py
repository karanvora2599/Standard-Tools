"""
Agentic Pair Trader — Google Gemini 2.0 Flash.
Gemini scans for cointegrated pairs and produces a trade plan with position sizing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import _header, _log, route_request, run_agent, setup_logging

# ── Configuration ──────────────────────────────────────────────────
GEMINI_API_KEY = ""  # Replace with your key
MODEL = "gemini-2.0-flash"

UNIVERSE = ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL"]
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"
ACCOUNT = 500_000.0

SYSTEM_PROMPT = """You are a quantitative pairs trader specialising in statistical arbitrage.

Your workflow:
1. Call scan_pairs on the provided universe to identify all cointegrated pairs
2. For the top cointegrated pairs (up to 5), call run_cointegration_test for each to get:
   - ADF statistic, p-value, hedge ratio, half-life
   - Current spread z-score and directional signal
3. For pairs with an active signal (z-score outside ±1.5), call get_position_size
   for each leg to determine shares and dollar risk
4. Produce a trade plan document that includes:
   - Summary table of all cointegrated pairs ranked by half-life (fastest first)
   - For each actionable pair: exact entry instruction (which leg long/short, hedge ratio)
   - Position sizes for each leg with dollar risk and stop placement
   - Risk notes: correlation risk, sector concentration, expected reversion time

Key concepts to explain in your report:
- Half-life tells us how fast the spread mean-reverts; shorter = more trades per year
- Z-score tells us how far the spread is from its mean; enter at ±2, exit at 0
- The hedge ratio tells us how many shares of B to short per share of A long

Use exact numbers from tool results in all recommendations."""

USER_REQUEST = f"""
Perform a full statistical arbitrage scan on the following sector universe:

Tickers    : {', '.join(UNIVERSE)}
Period     : {START_DATE} to {END_DATE}
Account    : ${ACCOUNT:,.0f}
Risk/trade : 1% of account per leg

Scan for cointegrated pairs, deep-dive the top 5, and produce an actionable
trade plan for any pair with a current signal (|z-score| > 1.5).
Include position sizes for both legs of each live trade.
""".strip()


if __name__ == "__main__":
    log_file = setup_logging("agent_pair_trader_gemini")

    _header("Agentic Pair Trader — Gemini 2.0 Flash")
    _log("Log file", str(log_file))
    _log("Universe", ", ".join(UNIVERSE))
    _log("Period", f"{START_DATE} → {END_DATE}")
    _log("Account", f"${ACCOUNT:,.0f}")

    routed_categories = route_request(USER_REQUEST, api_key=GEMINI_API_KEY, model=MODEL)

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=GEMINI_API_KEY,
        model=MODEL,
        max_iterations=20,
        categories=routed_categories,
    )

    _header("FINAL TRADE PLAN")
    print(result)
