"""
Agentic Pair Trader — Claude autonomously scans a sector universe for
cointegrated pairs, selects the most tradeable ones, and produces
a trade plan with position sizing for each live signal.

The agent:
  1. Scans all combinations in the universe via scan_pairs
  2. Deep-dives each cointegrated pair with run_cointegration_test
  3. Checks current z-score signals and half-life for each pair
  4. Sizes both legs using get_position_size with ATR-based risk management
  5. Delivers a ranked trade plan sorted by signal strength and half-life
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import setup_logging, run_agent, _header, _log

# ── Configuration ──────────────────────────────────────────────────
ANTHROPIC_API_KEY = ""   # Replace with your key
MODEL             = "claude-haiku-4-5"

# US energy sector — historically good for pairs trading
UNIVERSE   = ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL"]
START_DATE = "2021-01-01"
END_DATE   = "2024-12-31"
ACCOUNT    = 500_000.0

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
    log_file = setup_logging("agent_pair_trader")

    _header("Agentic Pair Trader — Claude Haiku")
    _log("Log file",  str(log_file))
    _log("Universe",  ", ".join(UNIVERSE))
    _log("Period",    f"{START_DATE} → {END_DATE}")
    _log("Account",   f"${ACCOUNT:,.0f}")

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=ANTHROPIC_API_KEY,
        model=MODEL,
        max_iterations=20,
    )

    _header("FINAL TRADE PLAN")
    print(result)
