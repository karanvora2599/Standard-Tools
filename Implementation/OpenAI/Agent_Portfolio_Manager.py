"""
Agentic Portfolio Manager — OpenAI / GPT-4o mini.
GPT autonomously reviews the portfolio and produces a rebalancing recommendation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import _header, _log, route_request, run_agent, setup_logging

# ── Configuration ──────────────────────────────────────────────────
OPENAI_API_KEY = ""  # Replace with your key
MODEL = "gpt-4o-mini"

PORTFOLIO = {
    "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "weights": [0.20, 0.20, 0.20, 0.20, 0.20],
}
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
BENCHMARK = "SPY"

SYSTEM_PROMPT = """You are a senior portfolio manager with expertise in quantitative risk management.

Your task is to perform a comprehensive portfolio review using the available tools.

Follow this workflow:
1. Use get_portfolio_analysis to get portfolio-level performance and correlation metrics
2. Use get_portfolio_risk_attribution to decompose risk (PCA, marginal contributions, factor loadings)
3. Use analyze_stock_risk for each individual holding to get beta, alpha, Sharpe, VaR
4. Use get_technical_analysis to check the current technical regime for each holding
5. Synthesize findings into a written report with:
   - Portfolio health summary (return, vol, drawdown vs benchmark)
   - Risk concentration alerts (which assets or factors dominate risk)
   - Individual position assessment (which holdings are dilutive vs accretive)
   - Specific rebalancing recommendations with reasoning

Be quantitative and specific. Reference exact numbers from the tool results."""

tickers_str = ", ".join(PORTFOLIO["tickers"])
weights_str = ", ".join(f"{w*100:.0f}%" for w in PORTFOLIO["weights"])

USER_REQUEST = f"""
Please perform a full portfolio review for the following equal-weight portfolio:

Tickers : {tickers_str}
Weights : {weights_str}
Period  : {START_DATE} to {END_DATE}
Benchmark: {BENCHMARK}

Use factor proxies SPY (market), IWM (size), IWD (value) for the factor regression.

Deliver a complete written analysis with:
- Portfolio performance vs benchmark
- Risk attribution and concentration
- Per-asset beta/alpha/Sharpe ranking
- Technical regime for each holding
- Rebalancing recommendation (keep / trim / add) for each position
""".strip()


if __name__ == "__main__":
    log_file = setup_logging("agent_portfolio_openai")

    _header("Agentic Portfolio Manager — GPT-4o mini")
    _log("Log file", str(log_file))
    _log("Portfolio", tickers_str)
    _log("Period", f"{START_DATE} → {END_DATE}")

    routed_categories = route_request(USER_REQUEST, api_key=OPENAI_API_KEY, model=MODEL)

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=OPENAI_API_KEY,
        model=MODEL,
        max_iterations=20,
        categories=routed_categories,
        # weights, sizing, capacity and stress are `portfolio`.
        # The router still narrows WITHIN the runtime; the runtime
        # is what makes the narrowing enforceable rather than
        # advisory. See Documentation/19_runtimes.md.
        registry="research+portfolio",
    )

    _header("FINAL REPORT")
    print(result)
