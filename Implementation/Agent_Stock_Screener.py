"""
Agentic Stock Screener — Claude autonomously filters a universe of stocks,
profiles every passing name, and produces a ranked watchlist with trade sizes.

The agent:
  1. Applies technical + fundamental filters via run_screener
  2. Runs get_technical_analysis for each passing stock to assess current momentum
  3. Runs analyze_stock_risk for each to get beta, alpha, Sharpe, VaR
  4. Determines optimal position size for each via get_position_size
  5. Optionally runs a quick backtest on the highest-conviction names
  6. Produces a ranked watchlist report with a buy/watch/avoid verdict per stock
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import setup_logging, run_agent, _header, _log

# ── Configuration ──────────────────────────────────────────────────
ANTHROPIC_API_KEY = ""  # Replace with your key
MODEL = "claude-haiku-4-5"

# Large-cap tech + growth universe
UNIVERSE = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "NFLX",
    "ADBE",
    "CRM",
    "NOW",
    "SNOW",
    "SHOP",
    "UBER",
    "ABNB",
    "PYPL",
    "AMD",
    "INTC",
    "QCOM",
    "TXN",
]
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
ACCOUNT = 500_000.0
MAX_POSITION = 0.10  # max 10% of account in any single name

SYSTEM_PROMPT = """You are a quantitative equity analyst responsible for building a momentum watchlist.

Your screening and analysis workflow:

Step 1 — Screen
  Call run_screener with these filters:
    rsi_min: 40          (avoid stocks in free-fall)
    rsi_max: 65          (avoid overbought names)
    price_above_sma: 50  (require uptrend)
    beta_max: 2.0        (avoid excessive vol)
    beta_min: 0.4        (require meaningful market correlation)
  Sort by rsi_14 ascending (most room to run first).

Step 2 — Enrich each passing stock
  For each stock that passes:
  a) Call get_technical_analysis with all indicators to get the full picture
  b) Call analyze_stock_risk (period="1y") to get alpha, beta, Sharpe, VaR

Step 3 — Size each position
  For each stock, call get_position_size with:
    risk_per_trade_pct: 0.01  (risk 1% of account)
    atr_period: 14
    atr_multiplier: 2.0

Step 4 — Backtest top 3 names
  For the 3 stocks with the best Sharpe ratio, call compare_strategies to find
  the best-performing strategy over the analysis period.

Step 5 — Write the watchlist report
  Rank all passing stocks from highest to lowest conviction.
  For each stock include: verdict (BUY / WATCH / AVOID), reasoning, key metrics,
  suggested position size, and stop-loss level (last close minus 2×ATR).

Be specific and quantitative. Show exact numbers from tool results."""

USER_REQUEST = f"""
Build a momentum watchlist from the following universe:

Universe : {', '.join(UNIVERSE)}
Period   : {START_DATE} to {END_DATE}
Account  : ${ACCOUNT:,.0f}  (max {MAX_POSITION*100:.0f}% per position)

Screen for quality momentum stocks, profile each passing name fully (technical +
risk + position size), backtest the top 3 names, and produce a ranked watchlist
report with a BUY / WATCH / AVOID verdict and stop-loss level for each stock.
""".strip()


if __name__ == "__main__":
    log_file = setup_logging("agent_stock_screener")

    _header("Agentic Stock Screener — Claude Haiku")
    _log("Log file", str(log_file))
    _log("Universe", f"{len(UNIVERSE)} stocks")
    _log("Period", f"{START_DATE} → {END_DATE}")
    _log("Account", f"${ACCOUNT:,.0f}")

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=ANTHROPIC_API_KEY,
        model=MODEL,
        max_iterations=25,
        # screening and profiling are both `research`.
        # A tool outside this runtime is refused by name rather than
        # run. See Documentation/19_runtimes.md.
        registry="research+backtest+portfolio",
    )

    _header("FINAL WATCHLIST")
    print(result)
