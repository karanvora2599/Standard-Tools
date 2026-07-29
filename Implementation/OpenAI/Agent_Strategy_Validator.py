"""
Agentic Strategy Validator — OpenAI / GPT-4o mini.
GPT runs a candidate strategy through a full pre-deployment validation
pipeline: backtest, robustness check, data-quality check, realistic
portfolio simulation, and capacity check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _agent_utils import _header, _log, route_request, run_agent, setup_logging

# ── Configuration ──────────────────────────────────────────────────
OPENAI_API_KEY = ""  # Replace with your key
MODEL = "gpt-4o-mini"

STRATEGY_SYMBOL = "AAPL"
STRATEGY_TYPE = "sma_crossover"
PARAM_GRID = {"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]}
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"

PORTFOLIO_TICKERS = ["AAPL", "MSFT"]
PORTFOLIO_WEIGHTS = [0.5, 0.5]
ACCOUNT_SIZE = 250_000.0

SYSTEM_PROMPT = """You are a quantitative strategy validator. Your job is to decide whether a
candidate strategy is actually ready for real capital — never trust a single
backtest number in isolation.

Your tools and when to use them:
- run_backtest_compact: run the candidate strategy once for a summary/risk/
  cost breakdown (same built-in strategies as run_sma_backtest etc., but
  returns compact sub-reports plus artifact URIs instead of the full equity
  curve/trade log inline — use this instead of run_sma_backtest here).
- get_robustness_diagnostics: run a parameter grid search around the
  candidate's parameters and report the Deflated Sharpe Ratio (has the best
  result survived correcting for having been selected as the best of many
  trials?), parameter sensitivity (how much worse is the median trial than
  the best?), and a block-bootstrap confidence interval on the best trial's
  Sharpe. This is a same-sample confidence check, NOT a substitute for
  out-of-sample walk-forward validation — it answers "how sure am I this
  number is real," not "would it have held up on unseen data."
- get_data_quality_report: check the underlying OHLCV for missing bars,
  stale/frozen prices, and large single-bar jumps, plus dataset provenance
  (adjusted / survivorship-free / point-in-time guarantees). Missing bars
  has no market-holiday calendar, so US holidays will appear as
  false-positive gaps — treat findings as leads to investigate, not proven
  defects. Run this BEFORE trusting the backtest/robustness results above.
- run_portfolio_simulation: simulate the validated strategy as a real
  shared-cash, multi-asset portfolio — one cash balance, positions sized
  against current equity, realistic commission/slippage, leverage limits.
  target_weights is {ticker: {date: weight}} — for a simple static
  allocation, use ONE rebalance date (the start date) per ticker with the
  requested weight for each; more rebalance dates only if the user asks for
  periodic rebalancing.
- get_capacity_report: given the same target_weights (as a single snapshot,
  not a full rebalance panel) and an account size, check whether any
  position would be too large relative to that ticker's own average daily
  trading volume — a backtest that looks great on paper can be untradeable
  at real size. Reports max supportable account size and days-to-liquidate.
- run_pair_trade_backtest: for a cointegrated PAIRS strategy instead of a
  single-ticker one — takes a hedge_ratio (typically from a prior
  run_cointegration_test) and backtests both legs as one synchronized,
  shared-cash trade. Use this instead of run_portfolio_simulation when the
  candidate strategy is a pairs trade, not a multi-asset allocation.

Workflow for a single-ticker/multi-asset strategy validation request:
1. get_data_quality_report on the underlying data first — flag any serious
   issue before proceeding
2. run_backtest_compact for the headline numbers
3. get_robustness_diagnostics to check the result isn't overfit
4. run_portfolio_simulation to see realistic shared-cash performance at the
   target allocation
5. get_capacity_report to confirm the target account size is actually
   tradeable given each ticker's liquidity

Produce a written go/no-go verdict citing exact numbers from every tool
call: headline return/Sharpe, Deflated Sharpe Ratio, any data-quality
findings, realistic portfolio-simulation performance after costs, and
maximum supportable account size. Be specific about what would make this
a "no-go" (e.g. DSR below 0.95, serious data-quality findings, capacity
far below the target account size)."""

tickers_str = ", ".join(PORTFOLIO_TICKERS)
weights_str = ", ".join(f"{w*100:.0f}%" for w in PORTFOLIO_WEIGHTS)

USER_REQUEST = f"""
I'm considering deploying an SMA crossover strategy on {STRATEGY_SYMBOL} and
want a full pre-deployment validation before committing capital.

Candidate strategy : {STRATEGY_TYPE} on {STRATEGY_SYMBOL}
Parameter grid      : fast_period in {PARAM_GRID['fast_period']}, slow_period in {PARAM_GRID['slow_period']}
Backtest period      : {START_DATE} to {END_DATE}

If it validates, I'd deploy it as part of a portfolio:
Tickers  : {tickers_str}
Weights  : {weights_str}
Account  : ${ACCOUNT_SIZE:,.0f}

Please:
1. Check the underlying data quality for {STRATEGY_SYMBOL} first
2. Run the compact backtest for the headline numbers
3. Run robustness diagnostics on the parameter grid — is this a real edge or
   an overfit parameter combination?
4. Simulate the {tickers_str} portfolio at the weights above with realistic
   costs
5. Check whether ${ACCOUNT_SIZE:,.0f} is actually tradeable given each
   ticker's liquidity

Give me a clear go/no-go verdict with exact numbers.
""".strip()


if __name__ == "__main__":
    log_file = setup_logging("agent_strategy_validator_openai")

    _header("Agentic Strategy Validator — GPT-4o mini")
    _log("Log file", str(log_file))
    _log("Candidate", f"{STRATEGY_TYPE} on {STRATEGY_SYMBOL}")
    _log("Portfolio", tickers_str)
    _log("Account", f"${ACCOUNT_SIZE:,.0f}")

    routed_categories = route_request(USER_REQUEST, api_key=OPENAI_API_KEY, model=MODEL)

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        api_key=OPENAI_API_KEY,
        model=MODEL,
        max_iterations=20,
        categories=routed_categories,
    )

    _header("VALIDATION VERDICT")
    print(result)
