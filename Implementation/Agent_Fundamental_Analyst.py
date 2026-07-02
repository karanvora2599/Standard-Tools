"""
Agentic Fundamental Analyst using Claude Haiku.
Combines fundamental data, parameter optimization, advanced indicators,
rolling beta, and extended risk metrics into a complete stock deep-dive.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import setup_logging, run_agent, _header, _log

# ── Configuration ──────────────────────────────────────────────────
ANTHROPIC_API_KEY = ""   # Replace with your key
MODEL             = "claude-haiku-4-5"

SYMBOLS    = ["AAPL", "MSFT", "NVDA"]
BENCHMARK  = "SPY"
START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"

SYSTEM_PROMPT = """You are a senior equity analyst conducting a deep-dive on a set of stocks.

Your workflow:

Step 1 — Fundamentals
  For each stock call get_stock_fundamentals.
  Note sector, PE ratios, P/B, debt/equity, ROE, profit margins, market cap.
  Flag any ratio that looks stretched or attractive vs peers.

Step 2 — Advanced technical signals
  For each stock call get_advanced_indicators (default periods are fine).
  Note the Parabolic SAR trend (bullish/bearish), Wilder ATR as % of price,
  and MFI signal (overbought/oversold/neutral).

Step 3 — Extended risk profile
  For each stock call get_extended_risk_metrics vs SPY.
  Compare Calmar ratio, Treynor ratio, and parametric VaR at 95/99.

Step 4 — Rolling beta drift
  For each stock call get_rolling_beta (window=60) vs SPY.
  Flag any stock where current_beta differs from beta_6m_ago by more than 0.2,
  or where beta_trend is not 'stable'.

Step 5 — Parameter optimisation
  For the stock with the best Calmar ratio call run_backtest_optimization:
    strategy: 'sma_crossover'
    param_grid: {'fast_period': [5, 10, 20], 'slow_period': [30, 50, 100]}
  Report the top 3 parameter combinations.

Step 6 — Write the research note
  ## FUNDAMENTAL SUMMARY TABLE
  ## TECHNICAL SIGNALS (SAR · ATR · MFI)
  ## EXTENDED RISK PROFILE
  ## ROLLING BETA ANALYSIS
  ## OPTIMAL STRATEGY PARAMETERS (best Calmar stock)
  ## FINAL RANKING & CONVICTION CALLS

Be specific: cite exact numbers from tool results throughout."""

symbols_str = ", ".join(SYMBOLS)

USER_REQUEST = f"""
Conduct a comprehensive fundamental + quantitative deep-dive on:

Stocks    : {symbols_str}
Benchmark : {BENCHMARK}
Period    : {START_DATE} to {END_DATE}

For each stock: fundamentals, advanced technical signals, extended risk metrics,
rolling beta vs SPY. Then optimise SMA crossover parameters for the highest-Calmar stock.
Produce a research note with a ranked conviction list (BUY / HOLD / AVOID).
""".strip()


if __name__ == "__main__":
    log_file = setup_logging("agent_fundamental_analyst")

    _header("Agentic Fundamental Analyst — Claude Haiku")
    _log("Log file",  str(log_file))
    _log("Stocks",    symbols_str)
    _log("Benchmark", BENCHMARK)
    _log("Period",    f"{START_DATE} → {END_DATE}")

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=ANTHROPIC_API_KEY,
        model=MODEL,
        max_iterations=25,
    )

    _header("RESEARCH NOTE")
    print(result)
