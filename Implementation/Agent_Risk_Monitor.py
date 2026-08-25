"""
Agentic Risk Monitor — Claude autonomously audits a portfolio's risk profile
and fires alerts for any metric that breaches defined thresholds.

The agent:
  1. Computes portfolio-level metrics and checks them against thresholds
  2. Runs PCA + factor attribution to detect hidden concentration
  3. Profiles each holding individually (beta, VaR, drawdown)
  4. Checks technical indicators for momentum deterioration signals
  5. Computes Hurst exponents to detect regime shifts in key holdings
  6. Produces a colour-coded risk report with ALERT / WARNING / OK status per metric
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import setup_logging, run_agent, _header, _log

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

# Risk thresholds — agent will flag any breach
THRESHOLDS = {
    "max_drawdown_pct": 15.0,  # alert if drawdown > 15%
    "portfolio_var95_pct": 3.0,  # alert if daily VaR95 > 3%
    "sharpe_ratio_min": 0.5,  # alert if Sharpe < 0.5
    "single_asset_risk_pct": 35.0,  # alert if any asset > 35% of portfolio variance
    "pc1_variance_pct": 65.0,  # alert if PC1 explains > 65% (hidden concentration)
    "beta_max": 1.5,  # alert if any holding has beta > 1.5
    "rsi_overbought": 70.0,  # alert if RSI > 70 on any holding
    "rsi_oversold": 30.0,  # alert if RSI < 30 on any holding
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
    log_file = setup_logging("agent_risk_monitor")

    _header("Agentic Risk Monitor — Claude Haiku")
    _log("Log file", str(log_file))
    _log("Portfolio", tickers_str)
    _log("Period", f"{START_DATE} → {END_DATE}")
    _log("Thresholds", str(THRESHOLDS))

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=ANTHROPIC_API_KEY,
        model=MODEL,
        max_iterations=25,
        # this agent profiles risk; it never runs a strategy.
        # A tool outside this runtime is refused by name rather than
        # run. See Documentation/19_runtimes.md.
        registry="research+portfolio",
    )

    _header("RISK AUDIT REPORT")
    print(result)
