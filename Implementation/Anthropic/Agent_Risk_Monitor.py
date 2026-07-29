"""
Agentic Risk Monitor — Anthropic / Claude Haiku.
Claude audits a portfolio's risk profile and fires alerts for threshold breaches.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import _header, _log, route_request, run_agent, setup_logging

# ── Configuration ──────────────────────────────────────────────────
ANTHROPIC_API_KEY = ""  # Replace with your key
MODEL = "claude-haiku-4-5"

PORTFOLIO = {
    "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"],
    "weights": [0.25, 0.25, 0.20, 0.20, 0.10],
}
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
BENCHMARK = "SPY"

THRESHOLDS = {
    "max_drawdown_pct": 15.0,
    "portfolio_var95_pct": 3.0,
    "sharpe_ratio_min": 0.5,
    "single_asset_risk_pct": 35.0,
    "pc1_variance_pct": 65.0,
    "beta_max": 1.5,
    "rsi_overbought": 70.0,
    "rsi_oversold": 30.0,
}

SYSTEM_PROMPT = """You are a risk officer performing a real-time portfolio risk audit.

Your job is to check a portfolio against predefined risk thresholds and flag breaches.

Follow this exact workflow:

1. Portfolio-level check
   Call get_portfolio_analysis. Flag if:
   - max_drawdown > {max_drawdown_pct}%       → ALERT
   - var_95 > {var95_pct}%                   → ALERT
   - sharpe_ratio < {sharpe_min}              → WARNING

2. Risk decomposition check
   Call get_portfolio_risk_attribution (with factor_tickers=[SPY,IWM,IWD]).
   Flag if:
   - Any single asset marginal risk contribution > {asset_risk_pct}%  → ALERT
   - PC1 explains > {pc1_pct}% of total variance                      → WARNING (concentration)
   - Market factor beta > 1.2                                          → WARNING

3. Per-asset check
   For each holding, call analyze_stock_risk. Flag if:
   - beta > {beta_max}                                                 → WARNING
   - max_drawdown > 30%                                                → ALERT
   - sharpe_ratio < 0                                                  → ALERT (negative risk-adj return)

4. Technical regime check
   For each holding, call get_technical_analysis. Flag if:
   - rsi_14 > {rsi_ob} (overbought)                                   → WARNING
   - rsi_14 < {rsi_os} (oversold / momentum collapse)                 → WARNING
   - price below sma_200                                               → WARNING (broken trend)

5. Regime check (Hurst) on largest 3 holdings
   Call run_hurst_analysis for the top 3 holdings by weight.
   Note regime shift if H drifts toward "random_walk" from a prior trend.

6. Write the risk report
   Use this structure:
   ## RISK SUMMARY
   Overall: HEALTHY / AT RISK / CRITICAL

   ## ALERTS (threshold breached — act now)
   ## WARNINGS (approaching threshold — monitor)
   ## OK (within thresholds)

   ## RECOMMENDATIONS
   Specific actions to reduce risk (trim, hedge, rebalance, add stop-losses)

Always cite exact metric values and the threshold that was breached.""".format(
    max_drawdown_pct=THRESHOLDS["max_drawdown_pct"],
    var95_pct=THRESHOLDS["portfolio_var95_pct"],
    sharpe_min=THRESHOLDS["sharpe_ratio_min"],
    asset_risk_pct=THRESHOLDS["single_asset_risk_pct"],
    pc1_pct=THRESHOLDS["pc1_variance_pct"],
    beta_max=THRESHOLDS["beta_max"],
    rsi_ob=THRESHOLDS["rsi_overbought"],
    rsi_os=THRESHOLDS["rsi_oversold"],
)

tickers_str = ", ".join(PORTFOLIO["tickers"])
weights_str = ", ".join(f"{w*100:.0f}%" for w in PORTFOLIO["weights"])

USER_REQUEST = f"""
Perform a full risk audit on the following portfolio:

Holdings : {tickers_str}
Weights  : {weights_str}
Period   : {START_DATE} to {END_DATE}
Benchmark: {BENCHMARK}

Risk thresholds to check against:
- Max drawdown threshold      : {THRESHOLDS['max_drawdown_pct']}%
- Portfolio daily VaR95 limit : {THRESHOLDS['portfolio_var95_pct']}%
- Minimum Sharpe ratio        : {THRESHOLDS['sharpe_ratio_min']}
- Max single-asset risk share : {THRESHOLDS['single_asset_risk_pct']}%
- PC1 concentration limit     : {THRESHOLDS['pc1_variance_pct']}%
- Individual beta limit       : {THRESHOLDS['beta_max']}
- RSI overbought level        : {THRESHOLDS['rsi_overbought']}
- RSI oversold level          : {THRESHOLDS['rsi_oversold']}

Produce a full risk audit report with ALERT / WARNING / OK status for each
metric and specific risk-reduction recommendations.
""".strip()


if __name__ == "__main__":
    log_file = setup_logging("agent_risk_monitor_anthropic")

    _header("Agentic Risk Monitor — Claude Haiku")
    _log("Log file", str(log_file))
    _log("Portfolio", tickers_str)
    _log("Period", f"{START_DATE} → {END_DATE}")
    _log("Thresholds", str(THRESHOLDS))

    routed_categories = route_request(
        USER_REQUEST, api_key=ANTHROPIC_API_KEY, model=MODEL
    )

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=ANTHROPIC_API_KEY,
        model=MODEL,
        max_iterations=25,
        categories=routed_categories,
    )

    _header("RISK AUDIT REPORT")
    print(result)
