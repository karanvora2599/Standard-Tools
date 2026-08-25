# Agent Tools

The agent module exposes every major library capability as **Pydantic-typed, LLM-callable functions**. Each tool takes one input model, does real computation, and returns one output model — both are directly JSON-serializable for function calling.

**Why use the agent module instead of calling the library directly?**

| Direct call | Agent tool |
|---|---|
| Returns `pd.DataFrame`, `pd.Series`, plain dicts | Returns Pydantic model — serializable without post-processing |
| Requires knowing the full library API | Single function per capability |
| Multiple imports, multiple calls | One import, one call |
| No schema for LLM | Schema auto-generated from Pydantic model |

---

## Quick Start

```python
from standard_quant_tools.agent import (
    # LLM wiring
    get_agent_tools, dispatch,
    # Backtest strategies
    run_sma_backtest, run_rsi_backtest, run_macd_backtest, run_bollinger_backtest,
    run_buy_and_hold, compare_strategies,
    # Analysis
    analyze_stock_risk, get_technical_analysis, get_portfolio_analysis, run_screener,
    run_factor_regression, run_cointegration_test, run_pca_analysis, run_hurst_analysis,
    # Advanced strategies
    run_regime_adaptive_backtest, scan_pairs, run_walk_forward_backtest,
    get_portfolio_risk_attribution, get_position_size,
    # Supplementary tools
    get_stock_fundamentals, run_backtest_optimization,
    get_advanced_indicators, get_rolling_beta, get_extended_risk_metrics,
    get_backtest_diagnostics,
)
from standard_quant_tools.agent import (
    BacktestInput, BuyAndHoldInput, CompareStrategiesInput,
    AnalysisInput, TechnicalInput, PortfolioInput, ScreenerInput,
    FactorRegressionInput, CointegrationInput, PCAInput, HurstInput,
    RegimeAdaptiveInput, PairScannerInput, WalkForwardInput,
    RiskAttributionInput, PositionSizerInput,
    FundamentalsInput, BacktestOptInput, AdvancedIndicatorsInput,
    RollingBetaInput, ExtendedRiskInput, BacktestDiagnosticsInput,
)

# Call any tool directly
result = analyze_stock_risk(AnalysisInput(symbol="AAPL"))
print(result.model_dump_json(indent=2))

# Or route an LLM tool call through dispatch (recommended for agent loops)
result = dispatch("analyze_stock_risk", {"symbol": "AAPL"})
print(result)  # plain dict, JSON-ready
```

---

## Tool Registry

`get_agent_tools()` returns **68 tool definitions** in the format both OpenAI and Anthropic expect. The schemas are derived automatically from Pydantic — no manual JSON authoring.

```python
from standard_quant_tools.agent import get_agent_tools

tools = get_agent_tools()
print(len(tools))  # 45

# Each tool follows the OpenAI function-calling format:
# {"type": "function", "function": {"name": ..., "description": ..., "parameters": <JSON Schema>}}
for t in tools:
    print(t["function"]["name"], "—", t["function"]["description"])
# run_sma_backtest      — SMA crossover backtest.
# run_rsi_backtest      — RSI mean-reversion backtest.
# run_macd_backtest     — MACD crossover backtest.
# run_bollinger_backtest — Bollinger Band mean-reversion backtest.
# run_buy_and_hold      — Buy-and-hold baseline: long the full period. Use as a passive benchmark.
# compare_strategies    — Run all four strategies on the same symbol and return ranked results vs buy-and-hold.
# analyze_stock_risk    — Full risk analysis: alpha, beta, Sharpe, VaR, CVaR.
# get_technical_analysis — Compute configurable technical indicators.
# get_portfolio_analysis — Multi-asset portfolio metrics.
# run_screener          — Filter a stock universe by fundamental and technical criteria.
# run_factor_regression — Multi-factor OLS regression: alpha, loadings, t-stats, p-values, R².
# run_cointegration_test — Engle-Granger cointegration: hedge ratio, half-life, spread z-score signal.
# run_pca_analysis      — PCA on multi-asset returns: explained variance, loadings, factor contributions.
# run_hurst_analysis    — Hurst exponent (DFA/R-S): regime classification and optional rolling breakdown.
# run_regime_adaptive_backtest — Classify market regime via Hurst, auto-select and optimise the best strategy.
# scan_pairs            — Scan a ticker universe for cointegrated pairs, ranked by half-life.
# run_walk_forward_backtest — Walk-forward validation: optimise in-sample, evaluate out-of-sample, return OOS stats.
# get_portfolio_risk_attribution — Deep portfolio risk decomposition: MCR per asset, PCA attribution, optional factor model.
# get_position_size     — ATR-based position sizing with optional Kelly criterion.
# get_stock_fundamentals — Fetch company metadata and key financial ratios (PE, P/B, debt/equity, ROE, market cap).
# run_backtest_optimization — Grid-search strategy parameters and return the top N combinations ranked by a chosen metric.
# get_advanced_indicators — Compute Parabolic SAR (trend), Wilder ATR (volatility), and MFI (volume-flow oscillator).
# get_rolling_beta      — Compute rolling OLS beta to detect beta drift over time vs a benchmark.
# get_extended_risk_metrics — Extended risk: Calmar ratio, Treynor ratio, parametric VaR 95/99, historical VaR 99, CVaR 99.
# run_custom_signal_backtest — Backtest a signal computed outside this library (your own alpha model) on one symbol.
# run_signal_panel_backtest — Backtest a pre-computed signal panel across a ticker universe, combined into portfolio metrics.
# run_regime_adaptive_walkforward_backtest — Leakage-free regime-adaptive backtest: regime/strategy/parameter selection per walk-forward window, evaluated strictly out-of-sample.
# get_backtest_diagnostics — Extended diagnostics for a built-in strategy: top drawdown episodes, trade expectancy/payoff/streaks with MAE/MFE, and exposure stats.
# run_portfolio_simulation — True shared-cash portfolio simulation with rebalancing at target-weight dates.
# run_pair_trade_backtest — Backtest a cointegrated pair as one synchronized two-leg trade sharing a single cash account.
# get_robustness_diagnostics — Same-sample robustness checks for a grid search: parameter sensitivity, Deflated Sharpe Ratio, block-bootstrap CI.
# get_capacity_report — How much account size a target-weight portfolio can support before positions outgrow each ticker's own trading volume.
# get_data_quality_report — Dataset provenance plus missing-bar/stale-price/price-jump detection on a symbol's OHLCV.
# run_backtest_compact — Compact backtest result: summary/risk/exposure/cost sub-reports plus equity-curve/trade-log artifact URIs.
# run_portfolio_optimization — Produce portfolio weights via Markowitz mean-variance, risk parity, or Black-Litterman.
# get_option_pricing    — Black-Scholes-Merton price and Greeks for a European option.
# get_implied_volatility — Solve for Black-Scholes-Merton implied volatility from an observed option price.
# (get_agent_tools() also includes get_volatility_estimators, get_correlation_analysis,
#  run_monte_carlo_simulation, run_stress_test, get_liquidity_metrics,
#  run_garch_volatility_forecast, run_kalman_hedge_ratio, and get_tail_risk_metrics
#  — see 09_advanced_agent_tools.md)

# Inspect the parameter schema for any tool:
import json
print(json.dumps(tools[0]["function"]["parameters"], indent=2))
```

### Dispatch — the recommended wiring approach

`dispatch(tool_name, arguments)` eliminates the manual `TOOL_FN` / `INPUT_MODEL` lookup. Pass the tool name and parsed arguments from any LLM response and get back a plain `dict` ready to send as a tool result.

```python
from standard_quant_tools.agent import dispatch

# Route any tool call in one line
result = dispatch("analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY"})
# → plain dict, same as result.model_dump()
```

Errors:
- **`ValueError`** — unknown tool name; message lists all 46 valid names.
- **`pydantic.ValidationError`** — arguments don't match the tool's input schema (bad types, missing required fields).

Every call through `dispatch()` can also produce an auditable decision record — inputs, data provenance, and an output hash, replayable later to check whether the result would still reproduce. See [10_auditability.md](10_auditability.md).

### Wiring up OpenAI

```python
import json
import openai
from standard_quant_tools.agent import get_agent_tools, dispatch

client   = openai.OpenAI()
messages = [{"role": "user", "content": "What is NVDA's current risk profile vs SPY?"}]

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=get_agent_tools(),
        tool_choice="auto",
    )
    msg = response.choices[0].message
    messages.append(msg)

    if not msg.tool_calls:
        print(msg.content)
        break

    for tc in msg.tool_calls:
        result = dispatch(tc.function.name, json.loads(tc.function.arguments))
        messages.append({
            "role":         "tool",
            "tool_call_id": tc.id,
            "content":      json.dumps(result),
        })
```

### Wiring up Anthropic

```python
import json
import anthropic
from standard_quant_tools.agent import get_agent_tools, dispatch

client   = anthropic.Anthropic()
messages = [{"role": "user", "content": "Screen mega-cap tech for RSI < 45, then backtest SMA 10/50 on any that pass"}]

while True:
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=4096,
        tools=get_agent_tools(),
        messages=messages,
    )

    if response.stop_reason != "tool_use":
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text)
        break

    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = dispatch(block.name, block.input)
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     json.dumps(result),
            })

    messages.append({"role": "user", "content": tool_results})
```

### Recommended system prompt

Tell the model what tools are available and how to use them together:

```python
SYSTEM = """
You are a quantitative analyst assistant with access to a 68-tool financial
toolkit. The 26 most commonly used are described below (see
09_advanced_agent_tools.md for the full list of the remaining 19 —
execution/diagnostic tools like run_regime_adaptive_walkforward_backtest,
get_backtest_diagnostics, run_portfolio_simulation, run_pair_trade_backtest,
get_robustness_diagnostics, get_capacity_report, get_data_quality_report,
run_backtest_compact, and the analytics/options tools like
get_volatility_estimators, run_garch_volatility_forecast,
get_tail_risk_metrics, get_option_pricing, and others):

CORE TOOLS (14)
1. run_sma_backtest / run_rsi_backtest / run_macd_backtest / run_bollinger_backtest
   — Backtest a single strategy on a symbol. Use at least 2 years of data.
     Prefer Sharpe > 1.0 and max drawdown < 30%.

2. run_buy_and_hold
   — Buy-and-hold baseline: holds long the entire period. Always use this as the
     passive benchmark when evaluating any active strategy.

3. compare_strategies
   — Runs all four strategies on the same symbol in one call and ranks them by the
     chosen metric (sharpe_ratio by default). Also includes buy_and_hold_return as
     the passive baseline. Use this instead of calling the four backtest tools
     individually when you want to compare approaches.

4. analyze_stock_risk
   — Full risk profile vs a benchmark: alpha, beta, Sharpe, Sortino, VaR, CVaR,
     Information Ratio. Use before recommending a position. Beta > 1.5 = amplified.

5. get_technical_analysis
   — Latest indicator snapshot: RSI, MACD, ADX, Bollinger, SMA, EMA, ATR, OBV,
     VWAP, Williams %R, Stochastic. RSI < 30 = oversold; ADX > 25 = strong trend.

6. get_portfolio_analysis
   — Weighted basket metrics + correlation matrix. Weights must sum to 1.0.
     Pairs near correlation 1.0 are redundant — poor diversification.

7. run_screener
   — Filter tickers by fundamental (PE, PB, ROE, margins) and technical (RSI, SMA,
     beta) criteria. Always screen first, then analyze or backtest the survivors.

8. run_factor_regression
   — Multi-factor OLS: alpha, loadings, t-stats, p-values, R², optional rolling.
     High R² = returns well explained by factors. Positive alpha = real edge.

9. run_cointegration_test
   — Engle-Granger test for a pair: p-value, hedge ratio, half-life, z-score signal.
     p < 0.05 and half-life < 30 bars = tradeable pair.

10. run_pca_analysis
    — PCA on multi-asset returns. PC1 is usually the market factor; later PCs capture
      sector or style tilts. Use to diagnose hidden concentration risk.

11. run_hurst_analysis
    — Hurst exponent (DFA or R/S). H > 0.55 = trending, 0.45–0.55 = random walk,
      H < 0.45 = mean-reverting. Choose strategy accordingly.

ADVANCED TOOLS (5)
12. run_regime_adaptive_backtest
    — Combines Hurst regime detection + parameter grid search in one call. Auto-selects
      the right strategy for the detected regime. Best for single-click strategy selection.

13. scan_pairs
    — Tests all O(n²/2) combinations in a ticker universe for cointegration. Returns
      top N pairs sorted by half-life (shortest = fastest mean reversion = most tradeable).
      Use before run_cointegration_test to narrow down candidates.

14. run_walk_forward_backtest
    — Gold-standard validation: optimise in-sample, test out-of-sample, repeat.
      avg_oos_sharpe > 0.5 and pct_windows_profitable > 60% = robust strategy.
      param_stability shows whether winning params are consistent (low overfitting).

15. get_portfolio_risk_attribution
    — Deep decomposition: Marginal Risk Contribution per asset (sum = 1.0), PCA
      variance attribution, optional multi-factor regression on the aggregate portfolio.
      Use when the user asks "what's driving my portfolio's risk?"

16. get_position_size
    — ATR-based stop-loss sizing. Optional Kelly criterion when win_rate/avg_win/
      avg_loss are known from a backtest. Always use this before sizing a real trade.

SUPPLEMENTARY TOOLS (5)
17. get_stock_fundamentals
    — Company metadata (sector, employees, country) and key financial ratios
      (forward/trailing PE, P/B, debt/equity, ROE, profit margins, market cap).
      Call before any fundamental analysis or screening.

18. run_backtest_optimization
    — Exhaustive parameter grid search for a single strategy. Returns top N
      combinations ranked by a chosen metric. Use this to find the best parameters
      before committing to a single run_*_backtest call.

19. get_advanced_indicators
    — Three indicators not in get_technical_analysis: Parabolic SAR (dynamic
      trailing-stop trend signal), Wilder ATR (true smoothed volatility), and
      MFI (volume-weighted RSI). SAR trend = "bullish"/"bearish"; MFI > 80 =
      overbought, < 20 = oversold.

20. get_rolling_beta
    — Rolling OLS beta in a sliding window to detect beta drift over time.
      Returns current beta plus 1m/3m/6m lookbacks and a trend label
      ("increasing"/"decreasing"/"stable"). Use alongside analyze_stock_risk
      (which gives a single static beta) for a fuller sensitivity picture.

21. get_extended_risk_metrics
    — Risk metrics not in analyze_stock_risk: CAGR, Calmar ratio (CAGR / |MDD|),
      Treynor ratio (excess return / beta), parametric VaR at 95% and 99%,
      historical VaR at 99%, and CVaR at 99%. Pair with analyze_stock_risk for
      a complete risk picture.

CUSTOM SIGNAL TOOLS (2)
22. run_custom_signal_backtest
    — Backtest a signal you (or an upstream system) computed outside this
      library — NOT one of run_sma_backtest / run_rsi_backtest / run_macd_backtest /
      run_bollinger_backtest. Pass signals as a {date: value} map (1=long, 0=flat,
      -1=short); this tool only backtests it, it never generates or second-guesses
      the signal logic. Use when the user supplies or references their own model's
      output rather than asking for a named indicator strategy.

23. run_signal_panel_backtest
    — Same idea as run_custom_signal_backtest but across a ticker universe: pass
      signal_panel as {ticker: {date: value}}. Returns per-ticker backtest results
      plus portfolio-level metrics (Sharpe, VaR, correlation-aware combination via
      weights). Use for a pre-computed cross-sectional signal instead of screening
      + backtesting each ticker one at a time.

WORKFLOW GUIDANCE
- Screen → analyze → backtest → size: the natural research chain.
- Use compare_strategies for a quick multi-strategy sweep; use run_buy_and_hold to
  establish the passive baseline when evaluating individual strategies.
- Use run_hurst_analysis before picking a strategy; or run_regime_adaptive_backtest
  to do both steps automatically.
- Use scan_pairs before run_cointegration_test to find the best pairs first.
- Use run_walk_forward_backtest after run_regime_adaptive_backtest to validate
  out-of-sample before committing capital.
- Use get_position_size last, after backtest statistics are known.
- Use get_stock_fundamentals early in any fundamental or screener workflow.
- Use run_backtest_optimization before a single backtest to identify the best params.
- Use get_advanced_indicators alongside get_technical_analysis for SAR/Wilder-ATR/MFI.
- Use get_rolling_beta when beta drift over time is relevant to the question.
- Use get_extended_risk_metrics alongside analyze_stock_risk for a complete risk view.
- Use run_custom_signal_backtest / run_signal_panel_backtest whenever the user
  provides or references their own signal — never substitute a built-in strategy
  for it, and never invent signal values yourself.

Always call tools — never guess numeric results.
"""
```

---

## Tool 1 — SMA Crossover Backtest

**When to use:** Testing trend-following strategies. Best on volatile assets with clear trends (individual stocks, sector ETFs). Works poorly on mean-reverting instruments.

**Signal logic:** Long when the fast SMA is above the slow SMA. Each bar where the fast SMA crosses above the slow SMA generates a new long entry; a cross below exits to flat.

```python
from standard_quant_tools.agent.tools import run_sma_backtest
from standard_quant_tools.agent.models import BacktestInput

result = run_sma_backtest(BacktestInput(
    symbol="TSLA",
    start_date="2020-01-01",
    end_date="2024-01-01",
    strategy_type="sma_crossover",
    parameters={"fast_period": 10, "slow_period": 50},
    initial_capital=10_000,
    commission_pct=0.001,   # 0.1% per trade side (typical retail broker)
    slippage_pct=0.0005,    # 0.05% slippage
))

print(f"Total Return   : {result.total_return:.1%}")
print(f"Sharpe Ratio   : {result.sharpe_ratio:.2f}")
print(f"Sortino Ratio  : {result.sortino_ratio:.2f}")
print(f"Max Drawdown   : {result.max_drawdown:.1%}")
print(f"Calmar Ratio   : {result.calmar_ratio:.2f}")
print(f"Win Rate       : {result.win_rate:.1%}")
print(f"Profit Factor  : {result.profit_factor:.2f}")
print(f"Num Trades     : {result.num_trades}")
print(f"Avg Trade P&L  : {result.avg_trade_return_pct:.2f}%")
print(f"Final Equity   : ${result.final_equity:,.2f}")
```

**Parameters:**

| Key | Type | Default | Description |
|---|---|---|---|
| `fast_period` | int | 10 | Fast SMA window (e.g. 10, 20) |
| `slow_period` | int | 30 | Slow SMA window (e.g. 50, 100, 200) |

**Common configurations:**

| Style | Fast | Slow | Notes |
|---|---|---|---|
| Swing | 10 | 30 | Reactive, more trades |
| Classic | 20 | 50 | Balanced |
| Position | 50 | 200 | Golden cross, fewer trades |

**Interpreting results:**

- `sharpe_ratio > 1.0` is good; `> 2.0` is excellent
- `max_drawdown` between −10% and −25% is typical for a trend-following system
- `calmar_ratio > 1.0` means the annual return exceeds the worst drawdown
- `profit_factor > 1.5` is a healthy edge; `< 1.0` means the strategy loses money
- Low `win_rate` (40–50%) is normal for trend strategies — a few large winners offset many small losses

---

## Tool 2 — RSI Mean Reversion Backtest

**When to use:** Testing mean-reversion strategies on liquid, range-bound instruments (broad market ETFs like SPY/QQQ, blue-chip dividend stocks). Works poorly on trending momentum stocks.

**Signal logic:** Enter long when RSI drops below `oversold`. Hold the position until RSI rises above `overbought`. Stateful — if RSI never reaches `overbought`, the position stays open.

```python
from standard_quant_tools.agent.tools import run_rsi_backtest
from standard_quant_tools.agent.models import BacktestInput

# Conservative: wait for deep oversold, exit quickly
result = run_rsi_backtest(BacktestInput(
    symbol="SPY",
    start_date="2018-01-01",
    end_date="2024-01-01",
    strategy_type="rsi_mean_reversion",
    parameters={"period": 14, "oversold": 30, "overbought": 60},
))

print(f"Win Rate       : {result.win_rate:.1%}")   # RSI strategies often have high win rates
print(f"Calmar Ratio   : {result.calmar_ratio:.2f}")
print(f"Profit Factor  : {result.profit_factor:.2f}")
print(f"Num Trades     : {result.num_trades}")
```

**Parameters:**

| Key | Type | Default | Description |
|---|---|---|---|
| `period` | int | 14 | RSI lookback (7 = sensitive, 21 = smooth) |
| `oversold` | float | 30 | Enter long threshold (lower = rarer trades) |
| `overbought` | float | 70 | Exit threshold (lower = shorter holds) |

**Common configurations:**

| Style | Period | Oversold | Overbought | Characteristics |
|---|---|---|---|---|
| Aggressive | 7 | 35 | 65 | Many trades, quick exits |
| Classic | 14 | 30 | 70 | Balanced frequency |
| Conservative | 21 | 25 | 60 | Rare trades, high conviction |

**Inspecting individual trades:**

```python
if result.trade_log:
    # Find best and worst trades
    sorted_trades = sorted(result.trade_log, key=lambda t: t.return_pct, reverse=True)
    print("Best trade:", sorted_trades[0])
    print("Worst trade:", sorted_trades[-1])

    # Average holding period
    from datetime import datetime
    holds = [
        (datetime.fromisoformat(t.exit_date) - datetime.fromisoformat(t.entry_date)).days
        for t in result.trade_log
        if t.exit_date and t.entry_date
    ]
    if holds:
        print(f"Avg holding period: {sum(holds)/len(holds):.0f} days")
```

---

## Tool 3 — MACD Crossover Backtest

**When to use:** Momentum-following on medium timeframes. Good for stocks with strong trends that develop over weeks. Better on volatile assets than SMA crossover because MACD reacts to acceleration, not just direction.

**Signal logic:** Long when the MACD line (fast EMA − slow EMA) is above the signal line (EMA of MACD). Flat otherwise. No stateful holding — the signal updates each bar.

```python
from standard_quant_tools.agent.tools import run_macd_backtest
from standard_quant_tools.agent.models import BacktestInput

result = run_macd_backtest(BacktestInput(
    symbol="QQQ",
    start_date="2019-01-01",
    end_date="2024-01-01",
    strategy_type="macd_crossover",
    parameters={"fast": 12, "slow": 26, "signal": 9},
))

# Compare against buy-and-hold
print(f"Strategy return: {result.total_return:.1%}")
print(f"Strategy Sharpe: {result.sharpe_ratio:.2f}")
print(f"Trades: {result.num_trades}")
```

**Parameters:**

| Key | Type | Default | Description |
|---|---|---|---|
| `fast` | int | 12 | Fast EMA (commonly 8 or 12) |
| `slow` | int | 26 | Slow EMA (commonly 21 or 26) |
| `signal` | int | 9 | Signal line EMA (commonly 9) |

**Less common but useful configurations:**

```python
# Faster MACD — more sensitive, more trades
parameters={"fast": 8, "slow": 21, "signal": 5}

# Weekly-equivalent MACD on daily bars
parameters={"fast": 5, "slow": 35, "signal": 5}
```

---

## Tool 4 — Bollinger Band Mean Reversion Backtest

**When to use:** Range-bound, oscillating instruments. Highly effective on commodity ETFs (GLD, USO), bond ETFs (TLT), and defensive sectors. Poor on strongly trending stocks.

**Signal logic:** Enter long when the closing price touches or crosses below the lower band. Exit when price returns to the middle band (the 20-day SMA). Position is held between these events regardless of how many bars it takes.

```python
from standard_quant_tools.agent.tools import run_bollinger_backtest
from standard_quant_tools.agent.models import BacktestInput

# Wider bands = rarer but higher-confidence entries
result = run_bollinger_backtest(BacktestInput(
    symbol="GLD",
    start_date="2015-01-01",
    end_date="2024-01-01",
    strategy_type="bollinger_reversion",
    parameters={"period": 20, "num_std": 2.0},
    initial_capital=50_000,
    commission_pct=0.0005,  # Lower commission for liquid ETF
))

print(f"Total Return : {result.total_return:.1%}")
print(f"Max Drawdown : {result.max_drawdown:.1%}")
print(f"Win Rate     : {result.win_rate:.1%}")  # Typically high for mean reversion
print(f"Num Trades   : {result.num_trades}")
```

**Parameters:**

| Key | Type | Default | Description |
|---|---|---|---|
| `period` | int | 20 | SMA window (also the basis for bands) |
| `num_std` | float | 2.0 | Band width in standard deviations |

**Band width effect on trading frequency:**

| `num_std` | Behaviour |
|---|---|
| 1.5 | More entries, shallower oversold |
| 2.0 | Classic setting |
| 2.5 | Rare but high-conviction entries |
| 3.0 | Extreme oversold only — very few trades |

---

## BacktestInput / BacktestResult — Full Reference

```python
from standard_quant_tools.agent.models import BacktestInput, BacktestResult, Trade

# All BacktestInput fields with descriptions:
inp = BacktestInput(
    symbol="AAPL",               # Required: ticker symbol
    start_date="2020-01-01",     # Required: ISO date string
    end_date="2024-01-01",       # Required: ISO date string
    strategy_type="sma_crossover",  # Required: strategy key (informational only)
    parameters={                 # Optional: strategy-specific params (see each tool)
        "fast_period": 10,
        "slow_period": 50,
    },
    initial_capital=10_000.0,    # Optional: default $10,000
    commission_pct=0.001,        # Optional: fraction per trade side (default 0.1%)
    slippage_pct=0.0005,         # Optional: fraction per trade side (default 0.05%)
    fill_price="close",          # Optional: "close" (default) or "next_open" — see below
)

# All BacktestResult fields:
result: BacktestResult = run_sma_backtest(inp)
result.total_return           # Net return as fraction (0.42 = +42%)
result.annualized_volatility  # Return std × √252
result.sharpe_ratio           # Annualized excess return / volatility
result.sortino_ratio          # Annualized excess return / downside volatility
result.max_drawdown           # Worst peak-to-trough (e.g. -0.23 = -23%)
result.calmar_ratio           # CAGR / |max_drawdown|; higher = better
result.win_rate               # Fraction of trades that were profitable (0–1)
result.profit_factor          # Gross profit / gross loss; > 1.5 is healthy
result.num_trades             # Number of completed round-trip trades
result.avg_trade_return_pct   # Average per-trade P&L in percent
result.final_equity           # Portfolio value at end date
result.equity_curve           # List[float] — daily portfolio value
result.trade_log              # Optional[List[Trade]] — per-trade details

# fill_price controls execution timing:
#   "close" (default) — signal known at bar t-1's close is filled at that same close.
#   "next_open" — more conservative; entries/exits/holds are priced off the bar's own
#                 Open where relevant (see run_strategy's docstring for the exact
#                 overnight/intraday decomposition).

# Quick benchmarking table — what's "good":
# Sharpe > 1.0     acceptable  |  > 2.0 excellent
# Max drawdown < 20%  comfortable  |  > 40% risky
# Win rate depends on strategy: 40% is fine for trend, 65%+ for mean reversion
# Profit factor > 1.5  good edge  |  < 1.0 losing strategy

# Serialize everything for LLM consumption:
import json
payload = result.model_dump()
payload.pop("equity_curve")  # Large list — omit if sending to LLM
print(json.dumps(payload, indent=2))
```

**Working with the equity curve:**

```python
import pandas as pd

# Reconstruct a dated equity curve for plotting
from standard_quant_tools.data.factory import DataFactory

provider = DataFactory.get_provider()
df = provider.get_ohlcv("AAPL", "2020-01-01", "2024-01-01")

result = run_sma_backtest(BacktestInput(
    symbol="AAPL", start_date="2020-01-01", end_date="2024-01-01",
    strategy_type="sma_crossover", parameters={"fast_period": 10, "slow_period": 50},
))

equity = pd.Series(result.equity_curve, index=df.index[:len(result.equity_curve)])
buy_hold = 10_000 * (1 + df["Close"].pct_change().fillna(0)).cumprod()

print(f"Strategy final  : ${equity.iloc[-1]:,.0f}")
print(f"Buy & hold final: ${buy_hold.iloc[-1]:,.0f}")
```

---

## Tool 5 — Buy-and-Hold Baseline

**When to use:** Always include buy-and-hold as the passive baseline before evaluating any active strategy. If an active strategy can't beat it, the extra risk isn't justified.

**Signal logic:** Enters long on the first bar and holds for the full period. One trade, zero signal lag. Commission and slippage are applied once on entry.

```python
from standard_quant_tools.agent import run_buy_and_hold, compare_strategies
from standard_quant_tools.agent import BuyAndHoldInput

result = run_buy_and_hold(BuyAndHoldInput(
    symbol="TSLA",
    start_date="2020-01-01",
    end_date="2024-01-01",
    initial_capital=10_000,
    commission_pct=0.001,
    slippage_pct=0.0005,
))

print(f"Total Return : {result.total_return:.1%}")
print(f"Sharpe Ratio : {result.sharpe_ratio:.2f}")
print(f"Max Drawdown : {result.max_drawdown:.1%}")
print(f"Final Equity : ${result.final_equity:,.0f}")
```

**Input:** `BuyAndHoldInput`

| Field | Type | Default | Description |
|---|---|---|---|
| `symbol` | str | required | Ticker symbol |
| `start_date` | str | required | ISO date (YYYY-MM-DD) |
| `end_date` | str | required | ISO date (YYYY-MM-DD) |
| `initial_capital` | float | `10_000.0` | Starting equity |
| `commission_pct` | float | `0.001` | One-time buy commission fraction |
| `slippage_pct` | float | `0.0005` | One-time buy slippage fraction |
| `fill_price` | str | `"close"` | `"close"` (default) or `"next_open"` — see `BacktestInput.fill_price` |

**Output:** `BacktestResult` — same schema as active strategy backtests. See the full reference above.

---

## Tool 6 — Compare Strategies

**When to use:** Get a side-by-side ranking of all four strategies in a single call instead of running each one individually. Includes the buy-and-hold return as a passive reference.

**How it works:** Runs SMA crossover, RSI mean reversion, MACD crossover, and Bollinger reversion on the same symbol and date range with a shared buy-and-hold baseline, then sorts all four by the chosen metric.

```python
from standard_quant_tools.agent import compare_strategies
from standard_quant_tools.agent import CompareStrategiesInput

result = compare_strategies(CompareStrategiesInput(
    symbol="AAPL",
    start_date="2020-01-01",
    end_date="2024-01-01",
    sort_by="sharpe_ratio",      # rank by Sharpe (default)
))

print(f"Best strategy     : {result.best_strategy}")
print(f"Buy-and-hold return: {result.buy_and_hold_return:.1%}")
for s in result.strategies:
    print(f"  {s.strategy:<22} sharpe={s.sharpe_ratio:.2f}  return={s.total_return:.1%}")
```

**Input:** `CompareStrategiesInput`

| Field | Type | Default | Description |
|---|---|---|---|
| `symbol` | str | required | Ticker symbol |
| `start_date` | str | required | ISO date (YYYY-MM-DD) |
| `end_date` | str | required | ISO date (YYYY-MM-DD) |
| `initial_capital` | float | `10_000.0` | Starting equity |
| `commission_pct` | float | `0.001` | Commission fraction per trade side |
| `slippage_pct` | float | `0.0005` | Slippage fraction per trade side |
| `sort_by` | str | `"sharpe_ratio"` | Metric to rank by: `total_return`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown` |
| `sma_parameters` | dict\|None | `None` | Override default SMA params |
| `rsi_parameters` | dict\|None | `None` | Override default RSI params |
| `macd_parameters` | dict\|None | `None` | Override default MACD params |
| `bollinger_parameters` | dict\|None | `None` | Override default Bollinger params |
| `fill_price` | str | `"close"` | `"close"` (default) or `"next_open"` — see `BacktestInput.fill_price` |

**Output:** `CompareStrategiesResult`

| Field | Type | Description |
|---|---|---|
| `symbol` | str | Ticker symbol |
| `sort_by` | str | Metric used for ranking |
| `best_strategy` | str | Name of the top-ranked strategy |
| `buy_and_hold_return` | float | Passive baseline total return |
| `strategies` | List[StrategyComparison] | All four strategies, sorted best first |

**`StrategyComparison` fields:** `strategy`, `parameters`, `total_return`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `win_rate`, `num_trades`, `final_equity`

**Custom parameter grids:**

```python
result = compare_strategies(CompareStrategiesInput(
    symbol="SPY",
    start_date="2019-01-01",
    end_date="2024-01-01",
    sort_by="calmar_ratio",
    sma_parameters={"fast_period": 20, "slow_period": 100},
    rsi_parameters={"period": 21, "oversold": 25, "overbought": 75},
))

best = result.strategies[0]
print(f"Winner: {best.strategy}, Calmar={best.calmar_ratio:.2f}, vs B&H={result.buy_and_hold_return:.1%}")
```

---

## Tool 7 — Stock Risk Analysis

**When to use:** Pre-trade due diligence, risk profiling for an LLM to explain a stock's characteristics. Compare multiple stocks on the same risk dimensions before building a portfolio.

```python
from standard_quant_tools.agent.tools import analyze_stock_risk
from standard_quant_tools.agent.models import AnalysisInput

result = analyze_stock_risk(AnalysisInput(
    symbol="NVDA",
    benchmark="SPY",   # Default. Use "QQQ" for tech comparison, "GLD" for commodity
    period="2y",       # "6mo", "1y", "2y", "3y", or "Nd" for N days
))

print(f"Alpha            : {result.alpha:.4f}")        # Daily excess return vs benchmark
print(f"Beta             : {result.beta:.2f}")          # < 1 = defensive, > 1 = amplified
print(f"R² (vs SPY)      : {result.r_squared:.2%}")    # How much variance is market-driven
print(f"Sharpe Ratio     : {result.sharpe_ratio:.2f}") # > 1.0 good, > 2.0 excellent
print(f"Sortino Ratio    : {result.sortino_ratio:.2f}")
print(f"Max Drawdown     : {result.max_drawdown:.1%}") # Negative number
print(f"VaR (95%, daily) : {result.var_95:.3%}")       # Max daily loss 95% of the time
print(f"CVaR (95%, daily): {result.cvar_95:.3%}")      # Expected loss when VaR is breached
print(f"Information Ratio: {result.information_ratio:.2f}")
```

**Interpreting each metric:**

| Metric | What it means | Typical range |
|---|---|---|
| `alpha` | Daily excess return above benchmark (raw, not annualized) | −0.002 to +0.002 |
| `beta` | Sensitivity to benchmark. 1.5 = stock moves 1.5× the market | 0.3 (utilities) to 2.5 (high-growth) |
| `r_squared` | How much of price movement is explained by the benchmark | 0.1 (idiosyncratic) to 0.9 (ETF-like) |
| `sharpe_ratio` | Risk-adjusted return (annualized) | 0.5–1.0 good for equities |
| `sortino_ratio` | Like Sharpe but only penalizes downside | Always ≥ Sharpe |
| `var_95` | Largest expected daily loss, 95% confidence | 0.01–0.05 (1%–5%) |
| `cvar_95` | Average loss on the 5% worst days | Always ≥ VaR |
| `information_ratio` | Active return vs benchmark / tracking error | > 0.5 strong |

**Comparing multiple stocks:**

```python
stocks  = ["AAPL", "NVDA", "TSLA", "MSFT", "JPM"]
results = [
    analyze_stock_risk(AnalysisInput(symbol=s, period="1y"))
    for s in stocks
]

# Build a comparison table
rows = [{
    "symbol":  r.symbol,
    "beta":    r.beta,
    "sharpe":  r.sharpe_ratio,
    "drawdown":f"{r.max_drawdown:.1%}",
    "var_95":  f"{r.var_95:.2%}",
    "ir":      r.information_ratio,
} for r in results]

import pandas as pd
print(pd.DataFrame(rows).sort_values("sharpe", ascending=False).to_string(index=False))
```

**Using a non-SPY benchmark:**

```python
# Tech-relative analysis: beats QQQ?
result_tech = analyze_stock_risk(AnalysisInput(symbol="NVDA", benchmark="QQQ", period="2y"))

# Inflation hedge: beats gold?
result_gold = analyze_stock_risk(AnalysisInput(symbol="GLD", benchmark="TLT", period="3y"))
```

---

## Tool 8 — Technical Analysis

**When to use:** Snapshot of current market conditions on a stock. Use to confirm an entry idea or quickly survey which signals are active. The LLM can interpret the `signals` dict and explain what it sees.

```python
from standard_quant_tools.agent.tools import get_technical_analysis
from standard_quant_tools.agent.models import TechnicalInput

result = get_technical_analysis(TechnicalInput(
    symbol="AAPL",
    start_date="2023-01-01",
    end_date="2024-06-01",
    indicators=["rsi", "macd", "bollinger", "adx", "sma", "obv", "atr"],
))

print(f"Last close  : ${result.last_close}")
print()
print("--- Indicator values ---")
for k, v in sorted(result.last_values.items()):
    print(f"  {k:<20}: {v}")
print()
print("--- Active signals ---")
for k, v in sorted(result.signals.items()):
    print(f"  {k:<30}: {v}")
```

**Available indicators:**

| Key | Values returned | Signals generated | Notes |
|---|---|---|---|
| `sma` | `sma_20`, `sma_50`, `sma_200` | `price_above_sma_20/50/200` | Trend filter |
| `ema` | `ema_12`, `ema_26` | — | Faster trend |
| `macd` | `macd`, `macd_signal`, `macd_histogram` | `macd_bullish` | Momentum |
| `rsi` | `rsi_14` | `rsi_oversold` (< 30), `rsi_overbought` (> 70) | Mean reversion |
| `stochastic` | `stoch_k`, `stoch_d` | `stoch_oversold` (K/D < 20) | Short-term momentum |
| `bollinger` | `bb_upper`, `bb_middle`, `bb_lower` | `price_near_lower_band`, `price_near_upper_band` | Volatility |
| `atr` | `atr_14` | — | Volatility in price units |
| `obv` | `obv` | `obv_rising` | Volume-price confirmation |
| `vwap` | `vwap` | `price_above_vwap` | Intraday fair value |
| `adx` | `adx`, `di_plus`, `di_minus` | `strong_trend` (ADX > 25), `bullish_di` (DI+ > DI−) | Trend strength |
| `williams_r` | `williams_r` | `williams_r_oversold` (< −80), `williams_r_overbought` (> −20) | Oscillator |

**Screening for entry conditions programmatically:**

```python
from standard_quant_tools.agent.tools import get_technical_analysis
from standard_quant_tools.agent.models import TechnicalInput

def is_oversold_in_trend(symbol: str, start: str, end: str) -> bool:
    """True when: RSI oversold + price above 50-day SMA + strong uptrend."""
    result = get_technical_analysis(TechnicalInput(
        symbol=symbol, start_date=start, end_date=end,
        indicators=["rsi", "sma", "adx"],
    ))
    s = result.signals
    return (
        s.get("rsi_oversold", False)
        and s.get("price_above_sma_50", False)
        and s.get("strong_trend", False)
        and s.get("bullish_di", False)
    )

candidates = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]
for ticker in candidates:
    if is_oversold_in_trend(ticker, "2023-01-01", "2024-01-01"):
        print(f"{ticker}: oversold pullback in a strong uptrend — potential entry")
```

**Building a multi-stock technical dashboard:**

```python
from standard_quant_tools.agent.tools import get_technical_analysis
from standard_quant_tools.agent.models import TechnicalInput
import pandas as pd

tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "META"]
rows = []

for ticker in tickers:
    r = get_technical_analysis(TechnicalInput(
        symbol=ticker,
        start_date="2023-06-01",
        end_date="2024-01-01",
        indicators=["rsi", "macd", "adx"],
    ))
    rows.append({
        "ticker":        ticker,
        "close":         r.last_close,
        "rsi_14":        r.last_values.get("rsi_14"),
        "adx":           r.last_values.get("adx"),
        "macd_bullish":  r.signals.get("macd_bullish"),
        "strong_trend":  r.signals.get("strong_trend"),
        "rsi_oversold":  r.signals.get("rsi_oversold"),
    })

df = pd.DataFrame(rows).set_index("ticker")
print(df.to_string())
```

---

## Tool 9 — Portfolio Analysis

**When to use:** Evaluate a weighted basket of assets together. Identify diversification quality via correlation, compare equal-weight vs custom weights, understand portfolio-level risk metrics.

```python
from standard_quant_tools.agent.tools import get_portfolio_analysis
from standard_quant_tools.agent.models import PortfolioInput

result = get_portfolio_analysis(PortfolioInput(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    weights=[0.30, 0.25, 0.20, 0.15, 0.10],  # Must sum to 1.0
    start_date="2022-01-01",
    end_date="2024-01-01",
    benchmark="SPY",                           # For Information Ratio
))

print(f"Annualized Return     : {result.annualized_return:.2%}")
print(f"Annualized Volatility : {result.annualized_volatility:.2%}")
print(f"Sharpe Ratio          : {result.sharpe_ratio:.2f}")
print(f"Sortino Ratio         : {result.sortino_ratio:.2f}")
print(f"Max Drawdown          : {result.max_drawdown:.2%}")
print(f"Calmar Ratio          : {result.calmar_ratio:.2f}")
print(f"VaR (95%, daily)      : {result.var_95:.3%}")
print(f"CVaR (95%, daily)     : {result.cvar_95:.3%}")
print(f"Information Ratio     : {result.information_ratio:.2f}")
print(f"Total Return          : {result.total_return:.2%}")
```

**Extracting the correlation matrix:**

```python
import pandas as pd

corr_dict = result.correlation_matrix
corr_df   = pd.DataFrame(corr_dict)

print("Correlation matrix:")
print(corr_df.round(2).to_string())

# Find the most and least correlated pairs
pairs = [
    (t1, t2, corr_df.loc[t1, t2])
    for i, t1 in enumerate(result.tickers)
    for t2 in result.tickers[i+1:]
]
pairs.sort(key=lambda x: x[2])
print(f"\nLowest correlation (best diversification): {pairs[0][0]} / {pairs[0][1]} = {pairs[0][2]:.2f}")
print(f"Highest correlation (most redundant)     : {pairs[-1][0]} / {pairs[-1][1]} = {pairs[-1][2]:.2f}")
```

**Comparing equal-weight vs custom weights:**

```python
from standard_quant_tools.agent.tools import get_portfolio_analysis
from standard_quant_tools.agent.models import PortfolioInput

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
n       = len(tickers)

# Equal weight
eq = get_portfolio_analysis(PortfolioInput(
    tickers=tickers,
    weights=[1/n] * n,
    start_date="2022-01-01",
    end_date="2024-01-01",
))

# Overweight NVDA (the best performer in the period)
custom = get_portfolio_analysis(PortfolioInput(
    tickers=tickers,
    weights=[0.15, 0.15, 0.15, 0.15, 0.40],
    start_date="2022-01-01",
    end_date="2024-01-01",
))

print(f"Equal-weight  Sharpe : {eq.sharpe_ratio:.2f} | Return: {eq.annualized_return:.2%}")
print(f"Custom-weight Sharpe : {custom.sharpe_ratio:.2f} | Return: {custom.annualized_return:.2%}")
```

**Defensive vs aggressive portfolio:**

```python
# Defensive: bonds + gold + dividend stocks
defensive = get_portfolio_analysis(PortfolioInput(
    tickers=["TLT", "GLD", "VYM", "KO", "JNJ"],
    weights=[0.30, 0.20, 0.20, 0.15, 0.15],
    start_date="2022-01-01",
    end_date="2024-01-01",
))

# Aggressive: concentrated growth tech
aggressive = get_portfolio_analysis(PortfolioInput(
    tickers=["NVDA", "TSLA", "META", "AMZN", "AMD"],
    weights=[0.30, 0.25, 0.20, 0.15, 0.10],
    start_date="2022-01-01",
    end_date="2024-01-01",
))

print(f"Defensive — Sharpe: {defensive.sharpe_ratio:.2f} | MDD: {defensive.max_drawdown:.1%}")
print(f"Aggressive — Sharpe: {aggressive.sharpe_ratio:.2f} | MDD: {aggressive.max_drawdown:.1%}")
```

---

## Tool 10 — Stock Screener

**When to use:** Starting a research workflow. Narrow a large universe down to candidates before spending compute on analysis or backtesting. All filters are applied concurrently — screening 50 tickers takes roughly the same time as screening 5.

```python
from standard_quant_tools.agent.tools import run_screener
from standard_quant_tools.agent.models import ScreenerInput
import json

result = run_screener(ScreenerInput(
    tickers=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN",
             "JPM", "BAC", "V", "MA", "UNH", "JNJ", "PG", "KO"],
    filters={
        "pe_ratio_max":      35,
        "profit_margin_min": 0.15,   # At least 15% net margin
        "rsi_max":           50,     # Not already overbought
        "market_cap_min":    50_000_000_000,  # $50B+
        "beta_max":          1.5,    # Not excessively volatile
    },
    sort_by="rsi_14",
    ascending=True,  # Most oversold first
))

print(f"Passed: {result.num_passed} / {len(result.tickers_passed)} tickers")
print(f"Survivors: {result.tickers_passed}")

# Send to LLM
print(json.dumps(result.model_dump(), indent=2))
```

**Distinguishing "didn't pass" from "couldn't be evaluated":** a ticker missing from `tickers_passed` isn't necessarily one that failed a filter. `result.failed_filters` (`ticker -> filter key`) names genuine rejections; `result.failed_tickers` (`ticker -> error message`) is a separate bucket for data-fetch/compute exceptions, so a broken fetch is never silently indistinguishable from a ticker that simply didn't meet the bar. When `n_workers > 1`, `result.failed_batches` also carries any worker-process error that prevented an entire batch from returning results.

**All supported filters:**

| Filter | Type | Description | Example |
|---|---|---|---|
| `pe_ratio_max` | float | Forward P/E upper bound | `25` |
| `pb_ratio_max` | float | Price-to-Book upper bound | `5.0` |
| `debt_equity_max` | float | Debt-to-Equity upper bound | `150` |
| `roe_min` | float | Return on Equity minimum (decimal) | `0.15` = 15% |
| `profit_margin_min` | float | Net profit margin minimum (decimal) | `0.10` = 10% |
| `div_yield_min` | float | Dividend yield minimum (decimal) | `0.02` = 2% |
| `market_cap_min` | int | Market cap minimum (USD) | `10_000_000_000` |
| `rsi_max` | float | RSI(14) upper bound | `40` = oversold screen |
| `rsi_min` | float | RSI(14) lower bound | `60` = momentum screen |
| `price_above_sma` | int | Close must be above SMA(N) | `50` |
| `price_below_sma` | int | Close must be below SMA(N) | `200` |
| `beta_max` | float | Beta vs SPY upper bound | `1.2` |
| `beta_min` | float | Beta vs SPY lower bound | `0.5` |

**Pre-built screen recipes:**

```python
# Value screen: cheap, profitable, low leverage
value = ScreenerInput(
    tickers=["AAPL", "MSFT", "GOOGL", "JPM", "BAC", "WMT", "KO", "JNJ"],
    filters={"pe_ratio_max": 20, "pb_ratio_max": 3.0, "roe_min": 0.15, "debt_equity_max": 100},
    sort_by="forward_pe", ascending=True,
)

# Momentum screen: already running, confirmed by SMA structure
# (filters accepts one price_above_sma value per call — 200 = golden-zone confirmation)
momentum = ScreenerInput(
    tickers=["NVDA", "META", "MSFT", "AAPL", "GOOGL", "AMD", "AVGO"],
    filters={"rsi_min": 55, "price_above_sma": 200},
    sort_by="rsi_14", ascending=False,
)

# Oversold quality: temporary weakness in strong businesses
oversold_quality = ScreenerInput(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "V", "MA", "UNH"],
    filters={"rsi_max": 40, "profit_margin_min": 0.15, "beta_max": 1.5, "market_cap_min": 50_000_000_000},
    sort_by="rsi_14", ascending=True,
)

# Defensive dividend: income + low volatility
dividend = ScreenerInput(
    tickers=["KO", "PEP", "JNJ", "PG", "MCD", "T", "VZ", "O"],
    filters={"div_yield_min": 0.025, "beta_max": 0.8, "debt_equity_max": 200},
    sort_by="dividend_yield", ascending=False,
)

# Run any of them:
result = run_screener(oversold_quality)
```

---

## Chaining Tools — Multi-Step Workflows

The real power of the agent module is composing tools into workflows. Here are three common patterns.

### Pattern 1: Screen → Analyze → Decide

```python
from standard_quant_tools.agent.tools import run_screener, analyze_stock_risk
from standard_quant_tools.agent.models import ScreenerInput, AnalysisInput

# Step 1: find oversold large-cap stocks
screen = run_screener(ScreenerInput(
    tickers=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
             "V", "MA", "UNH", "JNJ", "PG", "HD", "KO", "WMT"],
    filters={"rsi_max": 40, "profit_margin_min": 0.12, "market_cap_min": 100_000_000_000},
    sort_by="rsi_14", ascending=True,
))
print(f"Screened to: {screen.tickers_passed}")

# Step 2: risk-profile each survivor
if screen.tickers_passed:
    risk_profiles = [
        analyze_stock_risk(AnalysisInput(symbol=t, period="1y"))
        for t in screen.tickers_passed
    ]
    # Keep only low-beta (defensive) candidates
    candidates = [r for r in risk_profiles if r.beta < 1.0 and r.sharpe_ratio > 0.5]
    for r in sorted(candidates, key=lambda x: x.sharpe_ratio, reverse=True):
        print(f"{r.symbol}: Sharpe={r.sharpe_ratio:.2f}, Beta={r.beta:.2f}, MDD={r.max_drawdown:.1%}")
```

### Pattern 2: Screen → Backtest Each Survivor

```python
from standard_quant_tools.agent.tools import run_screener, run_rsi_backtest
from standard_quant_tools.agent.models import ScreenerInput, BacktestInput

universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
            "JPM", "V", "MA", "UNH", "JNJ", "PG"]

screen = run_screener(ScreenerInput(
    tickers=universe,
    filters={"rsi_max": 45, "pe_ratio_max": 40},
    sort_by="rsi_14", ascending=True,
))

if screen.tickers_passed:
    results = []
    for ticker in screen.tickers_passed:
        bt = run_rsi_backtest(BacktestInput(
            symbol=ticker,
            start_date="2021-01-01",
            end_date="2024-01-01",
            strategy_type="rsi_mean_reversion",
            parameters={"period": 14, "oversold": 30, "overbought": 65},
        ))
        results.append({
            "ticker":    ticker,
            "sharpe":    bt.sharpe_ratio,
            "win_rate":  bt.win_rate,
            "drawdown":  bt.max_drawdown,
            "trades":    bt.num_trades,
        })

    import pandas as pd
    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print(df.to_string(index=False))
```

### Pattern 3: Technical Analysis → Conditional Backtest

```python
from standard_quant_tools.agent.tools import get_technical_analysis, run_sma_backtest
from standard_quant_tools.agent.models import TechnicalInput, BacktestInput

ticker = "NVDA"

# Step 1: check current market structure
tech = get_technical_analysis(TechnicalInput(
    symbol=ticker,
    start_date="2023-01-01",
    end_date="2024-01-01",
    indicators=["adx", "rsi", "sma"],
))

is_trending = tech.signals.get("strong_trend", False)
above_200   = tech.signals.get("price_above_sma_200", False)

print(f"{ticker}: strong trend={is_trending}, above 200-SMA={above_200}")

# Step 2: choose strategy based on regime
if is_trending and above_200:
    print("Trend regime detected — backtesting SMA crossover")
    result = run_sma_backtest(BacktestInput(
        symbol=ticker, start_date="2021-01-01", end_date="2024-01-01",
        strategy_type="sma_crossover", parameters={"fast_period": 10, "slow_period": 50},
    ))
else:
    print("Range-bound regime detected — backtesting RSI mean reversion")
    from standard_quant_tools.agent.tools import run_rsi_backtest
    result = run_rsi_backtest(BacktestInput(
        symbol=ticker, start_date="2021-01-01", end_date="2024-01-01",
        strategy_type="rsi_mean_reversion", parameters={"period": 14, "oversold": 30, "overbought": 70},
    ))

print(f"Sharpe: {result.sharpe_ratio:.2f}, Trades: {result.num_trades}")
```

---

## Complete End-to-End Agent Loop

A production-ready loop that handles multiple tool calls per turn, serializes results, and terminates cleanly. Works with both OpenAI and Anthropic with minor changes.

```python
import json
import anthropic
from standard_quant_tools.agent import get_agent_tools, dispatch

# ── Agent loop ────────────────────────────────────────────────────────────────

SYSTEM = """
You are a quantitative investment analyst. You have access to a 68-tool
financial toolkit; the 27 covered here are the most commonly used (see
09_advanced_agent_tools.md for the remaining execution/diagnostic tools):

Core (14): run_sma_backtest, run_rsi_backtest, run_macd_backtest,
run_bollinger_backtest, run_buy_and_hold (passive baseline),
compare_strategies (rank all 4 strategies in one call),
analyze_stock_risk, get_technical_analysis, get_portfolio_analysis,
run_screener, run_factor_regression, run_cointegration_test,
run_pca_analysis, run_hurst_analysis.

Advanced (5): run_regime_adaptive_backtest (Hurst + grid search in one call),
scan_pairs (find cointegrated pairs in a universe), run_walk_forward_backtest
(OOS validation), get_portfolio_risk_attribution (MCR + PCA decomposition),
get_position_size (ATR stop-loss + optional Kelly sizing).

Supplementary (6): get_stock_fundamentals (PE, P/B, D/E, ROE, market cap),
run_backtest_optimization (exhaustive parameter grid search, top N combos),
get_advanced_indicators (Parabolic SAR, Wilder ATR, MFI),
get_rolling_beta (rolling OLS beta drift over time),
get_extended_risk_metrics (Calmar, Treynor, parametric VaR 95/99, CVaR 99),
get_backtest_diagnostics (drawdown episodes, trade expectancy/MAE-MFE, exposure stats).

Custom signal (2): run_custom_signal_backtest (backtest a signal the user/an
upstream model already computed — never one you invent), run_signal_panel_backtest
(same idea across a ticker universe, combined into portfolio metrics).

Guidelines:
- Screen first if a broad universe is mentioned.
- Use compare_strategies for multi-strategy comparison; run_buy_and_hold
  as the passive reference when evaluating any single active strategy.
- Use run_regime_adaptive_backtest for single-click regime detection + strategy.
- Use scan_pairs before run_cointegration_test to narrow candidates.
- Use run_walk_forward_backtest to validate before recommending capital deployment.
- Use get_position_size after a backtest to size the trade.
- Use get_stock_fundamentals early in any fundamental workflow.
- Use run_backtest_optimization to find best params before running a single backtest.
- Use get_extended_risk_metrics alongside analyze_stock_risk for a full risk picture.
- Use run_custom_signal_backtest / run_signal_panel_backtest whenever the user
  supplies their own signal — do not substitute a built-in strategy for it.
- Always use at least 2 years of data for backtests.
- Interpret all numbers — translate Sharpe ratios, drawdowns, and betas into
  plain English before responding.
"""


def run_agent(user_message: str, max_turns: int = 10) -> str:
    client   = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=4096,
            system=SYSTEM,
            tools=get_agent_tools(),
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  → Calling {block.name}({list(block.input.keys())})")
                try:
                    result  = dispatch(block.name, block.input)
                    result.pop("equity_curve", None)  # trim large list before sending
                    content = json.dumps(result)
                except Exception as e:
                    content = json.dumps({"error": str(e)})

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     content,
                })

        messages.append({"role": "user", "content": tool_results})

    return "Max turns reached."


# ── Example queries ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        # Core tools
        "Screen FAANG + NVDA for PE < 35 and RSI < 50. For any that pass, run an RSI mean-reversion backtest from 2021 to 2024 and tell me which had the best risk-adjusted return.",
        "Compare the risk profiles of NVDA and JNJ vs SPY over the past 2 years. Which is more suitable for a conservative portfolio?",
        "Analyze an equal-weight portfolio of AAPL, MSFT, GOOGL, and AMZN from 2022 to 2024. How diversified is it?",
        # Advanced tools
        "Detect TSLA's market regime and automatically run the best strategy for it from 2022 to 2024.",
        "Scan KO, PEP, MCD, YUM, SBUX for cointegrated pairs. What's the best pair to trade right now?",
        "Walk-forward validate an RSI mean-reversion strategy on SPY from 2016 to 2024. Is it robust?",
        "Decompose the risk in a portfolio of AAPL 30%, MSFT 25%, GOOGL 20%, JPM 15%, GLD 10%. Which asset is the biggest risk driver?",
        "I have a $100,000 account and want to trade NVDA risking 1% per trade. What's my position size?",
    ]

    for q in queries[:1]:  # Run one query as demonstration
        print(f"\nQuery: {q}\n{'─' * 60}")
        answer = run_agent(q)
        print(f"\nAnswer:\n{answer}")
```

---

## Tool 11 — Multi-Factor Regression

**When to use:** Decompose a stock's return into exposures to named risk factors. Use ticker proxies for factors (SPY = market, IWM = size, IWD = value, QQQ = growth). Alpha is the return unexplained by the factors — persistent positive alpha is a real edge.

```python
from standard_quant_tools.agent.tools import run_factor_regression
from standard_quant_tools.agent.models import FactorRegressionInput

result = run_factor_regression(FactorRegressionInput(
    symbol="AAPL",
    factor_tickers=["SPY", "IWM", "IWD"],
    factor_names=["market", "size", "value"],   # Human-readable labels
    start_date="2021-01-01",
    end_date="2024-01-01",
    rolling_window=60,    # Optional: last 20 rolling-OLS points
))

print(f"Alpha (daily)   : {result.alpha:.6f}")
print(f"R²              : {result.r_squared:.4f}")
print(f"Adj R²          : {result.adj_r_squared:.4f}")
print(f"Observations    : {result.n_obs}")
print()
print("Factor loadings:")
for f, loading in result.loadings.items():
    t = result.t_stats[f]
    p = result.p_values[f]
    print(f"  {f:<12}: {loading:+.4f}  (t={t:.2f}, p={p:.4f})")
print(f"  {'alpha':<12}: t={result.t_stats['alpha']:.2f}, p={result.p_values['alpha']:.4f}")

if result.rolling_alpha_tail:
    print(f"\nRolling alpha (last 20 points): {result.rolling_alpha_tail}")
```

**FactorRegressionInput fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `symbol` | str | Yes | Asset to analyse |
| `factor_tickers` | List[str] | Yes | Ticker proxies for each factor |
| `factor_names` | List[str] | No | Human-readable labels; defaults to `factor_tickers` |
| `start_date` | str | Yes | ISO date |
| `end_date` | str | Yes | ISO date |
| `rolling_window` | int | No | If set, return the last 20 bars of rolling loadings |

**Interpreting output:**

- **Alpha**: Daily excess return unexplained by factors. Positive and statistically significant (p < 0.05) = real edge.
- **Loading**: Sensitivity to each factor. AAPL loading of 1.2 on SPY means it moves 1.2× the market.
- **R²**: Fraction of return variance explained by all factors combined. Low R² = highly idiosyncratic.
- **Rolling loadings**: Track whether factor exposures are stable or have drifted over time.

**Fama-French 3-factor using ETF proxies:**

```python
from standard_quant_tools.agent.tools import run_factor_regression
from standard_quant_tools.agent.models import FactorRegressionInput

# SPY = market, IWM-SPY ≈ size premium, IWD = value
# Note: the tool takes raw tickers and computes returns internally;
# for SMB you typically pass IWM and SPY separately, then interpret the contrast
result = run_factor_regression(FactorRegressionInput(
    symbol="TSLA",
    factor_tickers=["SPY", "IWM", "IWD", "QQQ"],
    factor_names=["market", "small_cap", "value", "growth"],
    start_date="2020-01-01",
    end_date="2024-01-01",
    rolling_window=126,   # 6-month rolling window
))

# Interpret significance
print(f"Alpha (ann.): {result.alpha * 252:.2%}")
print(f"R²: {result.r_squared:.2%}  (how much is explained by these 4 factors)")
print()
sig_threshold = 0.05
for factor in result.factors:
    loading = result.loadings[factor]
    p       = result.p_values[factor]
    stars   = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    print(f"  {factor:<12}: {loading:+.3f}  p={p:.3f} {stars}")

# Rolling alpha: is it trending up or down?
if result.rolling_alpha_tail:
    tail = result.rolling_alpha_tail
    trend = "improving" if tail[-1] > tail[0] else "deteriorating"
    print(f"\nRolling alpha trend: {trend}")
    print(f"  First point: {tail[0]:+.6f}  Last: {tail[-1]:+.6f}")
```

**Comparing factor exposures across a universe:**

```python
from standard_quant_tools.agent.tools import run_factor_regression
from standard_quant_tools.agent.models import FactorRegressionInput
import pandas as pd

universe = ["AAPL", "NVDA", "JPM", "KO", "GLD"]
rows = []

for ticker in universe:
    r = run_factor_regression(FactorRegressionInput(
        symbol=ticker,
        factor_tickers=["SPY", "IWD", "TLT"],
        factor_names=["market", "value", "bond"],
        start_date="2022-01-01",
        end_date="2024-01-01",
    ))
    rows.append({
        "ticker":  ticker,
        "alpha":   f"{r.alpha * 252:.2%}",
        "market":  f"{r.loadings['market']:+.2f}",
        "value":   f"{r.loadings['value']:+.2f}",
        "bond":    f"{r.loadings['bond']:+.2f}",
        "r2":      f"{r.r_squared:.2%}",
    })

df = pd.DataFrame(rows).set_index("ticker")
print(df.to_string())
# KO and GLD typically show low market loading and positive bond loading
# NVDA shows high market beta and negative value loading (growth tilt)
# JPM often shows negative bond loading (banks hurt by falling rates)
```

**Detecting factor drift with rolling loadings:**

```python
from standard_quant_tools.agent.tools import run_factor_regression
from standard_quant_tools.agent.models import FactorRegressionInput

result = run_factor_regression(FactorRegressionInput(
    symbol="AAPL",
    factor_tickers=["SPY", "QQQ"],
    factor_names=["market", "growth"],
    start_date="2019-01-01",
    end_date="2024-01-01",
    rolling_window=252,   # 1-year rolling
))

if result.rolling_loadings_tail:
    # Last 20 quarterly snapshots of growth-factor loading
    growth_trail = result.rolling_loadings_tail.get("growth", [])
    print("Recent 20 rolling growth loadings:")
    for i, v in enumerate(growth_trail):
        bar = "█" * int(abs(v) * 20)
        print(f"  t-{20-i:02d}: {v:+.3f}  {bar}")
    if growth_trail:
        drift = growth_trail[-1] - growth_trail[0]
        print(f"\nLoading drift: {drift:+.3f}  "
              f"({'more growth-like' if drift > 0 else 'less growth-like'} over time)")
```

---

## Tool 12 — Cointegration Test

**When to use:** Pairs trading research. Test whether two assets share a long-run equilibrium so that deviations from it are mean-reverting and therefore tradeable. A short half-life (< 20 bars) makes the strategy practical.

```python
from standard_quant_tools.agent.tools import run_cointegration_test
from standard_quant_tools.agent.models import CointegrationInput

result = run_cointegration_test(CointegrationInput(
    symbol_a="KO",
    symbol_b="PEP",
    start_date="2020-01-01",
    end_date="2024-01-01",
    zscore_window=30,   # Rolling window for the trading signal
))

print(f"Cointegrated    : {result.cointegrated}")
print(f"P-value         : {result.p_value:.4f}   (< 0.05 = cointegrated)")
print(f"Hedge ratio     : {result.hedge_ratio:.4f}   (long 1 KO, short {result.hedge_ratio:.2f} PEP)")
print(f"ADF statistic   : {result.adf_statistic:.4f}")
print(f"Half-life       : {result.half_life_days:.1f} bars")
print(f"Spread mean     : {result.spread_mean:.4f}")
print(f"Spread std      : {result.spread_std:.4f}")
print(f"Current z-score : {result.current_zscore:.2f}")
print(f"Signal          : {result.signal}")
# Signal values: "long_a_short_b" | "short_a_long_b" | "neutral"

print("\nCritical values:")
for level, cv in result.critical_values.items():
    print(f"  {level}: {cv:.4f}")
```

**CointegrationInput fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `symbol_a` | str | — | First asset (the "long" leg by convention) |
| `symbol_b` | str | — | Second asset (the "short" leg by convention) |
| `start_date` | str | — | ISO date |
| `end_date` | str | — | ISO date |
| `zscore_window` | int | 30 | Rolling window for the live z-score signal |

**Signal logic:**

| Z-score | Signal | Interpretation |
|---|---|---|
| < −2.0 | `long_a_short_b` | Spread is cheap — buy A, sell B |
| > +2.0 | `short_a_long_b` | Spread is expensive — sell A, buy B |
| −2 to +2 | `neutral` | Within normal range — hold or flat |

**Screening a candidate list for pairs:**

```python
from itertools import combinations
from standard_quant_tools.agent.tools import run_cointegration_test
from standard_quant_tools.agent.models import CointegrationInput

candidates = ["KO", "PEP", "MCD", "YUM"]
pairs = []

for a, b in combinations(candidates, 2):
    r = run_cointegration_test(CointegrationInput(
        symbol_a=a, symbol_b=b,
        start_date="2021-01-01", end_date="2024-01-01",
    ))
    if r.cointegrated and r.half_life_days < 30:
        pairs.append((a, b, r.p_value, r.half_life_days))

pairs.sort(key=lambda x: x[2])
for a, b, p, hl in pairs:
    print(f"{a}/{b}: p={p:.4f}, half-life={hl:.1f} bars")
```

**Full pairs trading decision framework:**

```python
from standard_quant_tools.agent.tools import run_cointegration_test
from standard_quant_tools.agent.models import CointegrationInput

def evaluate_pair(symbol_a: str, symbol_b: str,
                  start: str, end: str,
                  min_history_bars: int = 252) -> None:
    r = run_cointegration_test(CointegrationInput(
        symbol_a=symbol_a, symbol_b=symbol_b,
        start_date=start, end_date=end,
        zscore_window=30,
    ))

    print(f"\n{'─' * 50}")
    print(f"Pair: {symbol_a} / {symbol_b}")
    print(f"{'─' * 50}")

    # 1. Is it statistically cointegrated?
    if not r.cointegrated:
        print(f"✗ NOT cointegrated (p={r.p_value:.4f}) — do not trade this pair")
        return

    print(f"✓ Cointegrated  p={r.p_value:.4f}  ADF={r.adf_statistic:.3f}")
    print(f"  Critical values: 1%={r.critical_values['1%']:.3f}  "
          f"5%={r.critical_values['5%']:.3f}  10%={r.critical_values['10%']:.3f}")

    # 2. Is mean-reversion speed practical?
    if r.half_life_days > 60:
        print(f"⚠  Half-life {r.half_life_days:.0f} bars — slow reversion, needs patient sizing")
    elif r.half_life_days < 5:
        print(f"⚠  Half-life {r.half_life_days:.0f} bars — too fast, transaction costs will dominate")
    else:
        print(f"✓ Half-life {r.half_life_days:.0f} bars — good mean-reversion speed")

    # 3. Is there enough data?
    if r.n_obs < min_history_bars:
        print(f"⚠  Only {r.n_obs} observations — results may not be reliable")

    # 4. Current signal
    z = r.current_zscore
    print(f"\nCurrent z-score: {z:+.2f}  →  signal: {r.signal}")
    print(f"Hedge ratio: {r.hedge_ratio:.4f}  (long 1 {symbol_a}, short {r.hedge_ratio:.2f} {symbol_b})")
    print(f"Spread stats: mean={r.spread_mean:.4f}, std={r.spread_std:.4f}")

    if r.signal == "long_a_short_b":
        print(f"\n→ ACTION: Buy {symbol_a}, Sell {symbol_b}")
        print(f"  Entry: z-score {z:.2f} (below -2 threshold)")
        print(f"  Exit when z-score returns to 0")
    elif r.signal == "short_a_long_b":
        print(f"\n→ ACTION: Sell {symbol_a}, Buy {symbol_b}")
        print(f"  Entry: z-score {z:.2f} (above +2 threshold)")
        print(f"  Exit when z-score returns to 0")
    else:
        print(f"\n→ HOLD / FLAT: z-score within normal range ({z:.2f})")

# Example calls
evaluate_pair("KO", "PEP",  "2020-01-01", "2024-01-01")
evaluate_pair("GLD", "SLV", "2020-01-01", "2024-01-01")
evaluate_pair("AAPL", "TSLA", "2020-01-01", "2024-01-01")  # Likely NOT cointegrated
```

**Monitoring an active pairs position over time:**

```python
from standard_quant_tools.agent.tools import run_cointegration_test
from standard_quant_tools.agent.models import CointegrationInput

# Re-run monthly to check if the relationship still holds
history_periods = [
    ("2021-01-01", "2022-01-01"),
    ("2021-07-01", "2022-07-01"),
    ("2022-01-01", "2023-01-01"),
    ("2022-07-01", "2023-07-01"),
    ("2023-01-01", "2024-01-01"),
]

print("KO/PEP rolling cointegration stability:")
print(f"{'Period':<25} {'p-value':<10} {'Half-life':<12} {'Hedge ratio':<12} {'Z-score'}")
print("─" * 75)

for start, end in history_periods:
    r = run_cointegration_test(CointegrationInput(
        symbol_a="KO", symbol_b="PEP",
        start_date=start, end_date=end,
    ))
    coint_str = "✓" if r.cointegrated else "✗"
    print(f"{start}→{end}  {coint_str} p={r.p_value:.3f}   "
          f"hl={r.half_life_days:5.1f}d    hr={r.hedge_ratio:.3f}       z={r.current_zscore:+.2f}")
# If p-value spikes or hedge ratio drifts significantly, recalibrate or exit the position
```

---

## Tool 13 — PCA Analysis

**When to use:** Understand the hidden structure of a multi-asset portfolio. PCA decomposes correlated returns into orthogonal factors ordered by explained variance. PC1 is almost always the broad market; PC2 often captures a growth-vs-value tilt or sector contrast. Use this to identify latent risks and assess true diversification.

```python
from standard_quant_tools.agent.tools import run_pca_analysis
from standard_quant_tools.agent.models import PCAInput

result = run_pca_analysis(PCAInput(
    tickers=["AAPL", "MSFT", "GOOGL", "NVDA", "META",
             "JPM", "BAC", "GS", "TLT", "GLD"],
    start_date="2021-01-01",
    end_date="2024-01-01",
    n_components=3,
))

print(f"Assets : {result.tickers}")
print(f"Obs    : {result.n_obs}")
print()
print("Explained variance:")
for pc, evr in result.explained_variance_ratio.items():
    cumvar = result.cumulative_variance_ratio[pc]
    print(f"  {pc}: {evr:.1%}  (cumulative: {cumvar:.1%})")
print()
print("Factor loadings (eigenvectors):")
for pc in result.loadings:
    top = sorted(result.loadings[pc].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    print(f"  {pc}: " + ", ".join(f"{t}={v:+.3f}" for t, v in top))
print()
print("Per-asset factor contributions (marginal R²):")
for ticker, contribs in result.factor_contributions.items():
    total = sum(contribs.values())
    print(f"  {ticker}: PC1={contribs['PC1']:.3f}, PC2={contribs['PC2']:.3f}, PC3={contribs['PC3']:.3f}  (total={total:.3f})")
```

**PCAInput fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `tickers` | List[str] | — | Universe to decompose |
| `start_date` | str | — | ISO date |
| `end_date` | str | — | ISO date |
| `n_components` | int | 3 | Number of PCs to extract (≤ number of tickers) |

**Interpreting output:**

- **`explained_variance_ratio`**: Fraction of total return variance each PC captures. A PC1 of 0.60+ means a single market factor dominates — the portfolio is concentrated.
- **`cumulative_variance_ratio`**: Cumulative coverage. If PC1 + PC2 covers 90%, a 2-factor model is sufficient.
- **`loadings`**: Each column is an eigenvector (the PC). High loadings on many tech stocks → PC = "tech factor". Opposite signs → contrast factor (growth vs value).
- **`factor_contributions`**: How much of each asset's return variance each PC explains. An asset with low PC1 contribution is largely driven by idiosyncratic factors.

**Diagnosing hidden concentration in a portfolio:**

```python
from standard_quant_tools.agent.tools import run_pca_analysis
from standard_quant_tools.agent.models import PCAInput

# Seemingly diversified portfolio
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
           "NVDA", "AMD", "AVGO", "QCOM", "INTC"]

result = run_pca_analysis(PCAInput(
    tickers=tickers,
    start_date="2022-01-01",
    end_date="2024-01-01",
    n_components=3,
))

pc1_pct = result.explained_variance_ratio["PC1"]
cum2    = result.cumulative_variance_ratio.get("PC2", 0)

print(f"PC1 explains {pc1_pct:.0%} of all variance")
if pc1_pct > 0.50:
    print("⚠  A single market/sector factor dominates — this is NOT well diversified")
elif pc1_pct > 0.35:
    print("⚠  Moderate concentration — consider adding uncorrelated assets")
else:
    print("✓ Variance is spread across factors — reasonable diversification")

print(f"\nPC1 + PC2 explain {cum2:.0%} of all variance")

# Which assets are most "market-driven" (high PC1 contribution)?
print("\nAssets most exposed to PC1 (market factor):")
pc1_contribs = {t: result.factor_contributions[t]["PC1"] for t in tickers}
for ticker, contrib in sorted(pc1_contribs.items(), key=lambda x: -x[1])[:5]:
    print(f"  {ticker}: {contrib:.1%} of its variance explained by PC1")

# Which assets add the most genuine diversification (low PC1 contribution)?
print("\nMost idiosyncratic assets (low PC1, best diversifiers):")
for ticker, contrib in sorted(pc1_contribs.items(), key=lambda x: x[1])[:3]:
    print(f"  {ticker}: only {contrib:.1%} of variance from PC1")
```

**Comparing a concentrated vs diversified portfolio:**

```python
from standard_quant_tools.agent.tools import run_pca_analysis
from standard_quant_tools.agent.models import PCAInput

# Concentrated: all tech
tech_only = run_pca_analysis(PCAInput(
    tickers=["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
    start_date="2022-01-01", end_date="2024-01-01",
))

# Diversified: tech + bonds + commodities + financials
diversified = run_pca_analysis(PCAInput(
    tickers=["AAPL", "MSFT", "TLT", "GLD", "JPM", "XOM"],
    start_date="2022-01-01", end_date="2024-01-01",
))

print("Concentrated (all tech):")
print(f"  PC1 explains: {tech_only.explained_variance_ratio['PC1']:.0%}")
print(f"  PC1+PC2:      {tech_only.cumulative_variance_ratio.get('PC2', 0):.0%}")

print("\nDiversified (multi-asset):")
print(f"  PC1 explains: {diversified.explained_variance_ratio['PC1']:.0%}")
print(f"  PC1+PC2:      {diversified.cumulative_variance_ratio.get('PC2', 0):.0%}")

# Identifying what PC2 captures (often growth-vs-value or bond contrast)
print("\nDiversified portfolio — PC2 top loadings:")
pc2_loads = diversified.loadings["PC2"]
for t, v in sorted(pc2_loads.items(), key=lambda x: -abs(x[1])):
    bar = "+" * int(v * 20) if v > 0 else "-" * int(abs(v) * 20)
    print(f"  {t:<6}: {v:+.3f}  {bar}")
# Opposite signs between GLD/TLT and tech stocks → PC2 = flight-to-safety vs risk
```

**Finding which assets to add for better diversification:**

```python
from standard_quant_tools.agent.tools import run_pca_analysis
from standard_quant_tools.agent.models import PCAInput

# Existing portfolio
existing = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
# Candidates to add
candidates = ["TLT", "GLD", "XOM", "JPM", "WMT", "UNH"]

# Baseline
base = run_pca_analysis(PCAInput(tickers=existing,
                                  start_date="2022-01-01", end_date="2024-01-01"))
base_pc1 = base.explained_variance_ratio["PC1"]

print(f"Baseline PC1 concentration: {base_pc1:.0%}\n")
print("Effect of adding each candidate:")

for candidate in candidates:
    test = run_pca_analysis(PCAInput(
        tickers=existing + [candidate],
        start_date="2022-01-01", end_date="2024-01-01",
    ))
    new_pc1 = test.explained_variance_ratio["PC1"]
    delta   = new_pc1 - base_pc1
    arrow   = "↓ better" if delta < -0.02 else ("↑ worse" if delta > 0.01 else "≈ neutral")
    print(f"  +{candidate}: PC1={new_pc1:.0%}  ({delta:+.1%})  {arrow}")
```

---

## Tool 14 — Hurst Exponent

**When to use:** Regime detection before strategy selection. A trending regime (H > 0.55) favours momentum and trend-following. A mean-reverting regime (H < 0.45) favours contrarian strategies like RSI mean-reversion or Bollinger Band reversion. The rolling Hurst tracks how the regime evolves over time.

```python
from standard_quant_tools.agent.tools import run_hurst_analysis
from standard_quant_tools.agent.models import HurstInput

result = run_hurst_analysis(HurstInput(
    symbol="NVDA",
    start_date="2021-01-01",
    end_date="2024-01-01",
    method="dfa",           # "dfa" (default) or "rs"
    rolling_window=252,     # Optional: 1-year rolling Hurst
))

print(f"Symbol          : {result.symbol}")
print(f"Hurst exponent  : {result.hurst:.4f}")
print(f"Regime          : {result.regime}")
print(f"Fit R²          : {result.fit_r_squared:.4f}")
print(f"Method          : {result.method}")
print(f"Observations    : {result.n_obs}")

if result.rolling_current is not None:
    print(f"\nRolling Hurst (current)   : {result.rolling_current:.4f}")
    fracs = result.rolling_regime_fractions
    print(f"Trending fraction         : {fracs['trending']:.1%}")
    print(f"Random walk fraction      : {fracs['random_walk']:.1%}")
    print(f"Mean-reverting fraction   : {fracs['mean_reverting']:.1%}")
```

**HurstInput fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `symbol` | str | — | Ticker symbol |
| `start_date` | str | — | ISO date |
| `end_date` | str | — | ISO date |
| `method` | `Literal["dfa","rs"]` | `"dfa"` | `"dfa"` (Detrended Fluctuation Analysis) or `"rs"` (Rescaled Range) — any other value is rejected at input validation with a Pydantic error, not silently treated as `"rs"` |
| `rolling_window` | int | None | If set, compute rolling Hurst and return regime fractions |

**Regime table:**

| H value | Regime | Strategy implication |
|---|---|---|
| > 0.55 | `trending` | Use SMA crossover or MACD — momentum persists |
| 0.45–0.55 | `random_walk` | No persistent edge; reduce position sizing |
| < 0.45 | `mean_reverting` | Use RSI mean-reversion or Bollinger Band reversion |

**DFA vs R/S:**

DFA (default) is unbiased for realistic sample sizes — iid returns produce H ≈ 0.50. R/S has an upward bias, giving H ≈ 0.58 for iid returns, which can misclassify a random walk as trending. Use R/S only to compare with published H estimates that used R/S.

**Regime-conditional strategy selection:**

```python
from standard_quant_tools.agent.tools import run_hurst_analysis, run_sma_backtest, run_rsi_backtest
from standard_quant_tools.agent.models import HurstInput, BacktestInput

ticker = "TSLA"
hurst = run_hurst_analysis(HurstInput(
    symbol=ticker, start_date="2022-01-01", end_date="2024-01-01",
))

print(f"{ticker}: H={hurst.hurst:.3f}, regime={hurst.regime}")

if hurst.regime == "trending":
    result = run_sma_backtest(BacktestInput(
        symbol=ticker, start_date="2022-01-01", end_date="2024-01-01",
        strategy_type="sma_crossover", parameters={"fast_period": 10, "slow_period": 50},
    ))
    print("→ SMA crossover (trend regime)")
elif hurst.regime == "mean_reverting":
    result = run_rsi_backtest(BacktestInput(
        symbol=ticker, start_date="2022-01-01", end_date="2024-01-01",
        strategy_type="rsi_mean_reversion", parameters={"period": 14, "oversold": 30, "overbought": 70},
    ))
    print("→ RSI mean-reversion (mean-reverting regime)")
else:
    print("→ Random walk: no statistical edge; skip backtesting")
    result = None

if result:
    print(f"Sharpe: {result.sharpe_ratio:.2f} | MDD: {result.max_drawdown:.1%} | Trades: {result.num_trades}")
```

**Multi-stock universe regime scan:**

```python
from standard_quant_tools.agent.tools import run_hurst_analysis
from standard_quant_tools.agent.models import HurstInput

universe = ["AAPL", "MSFT", "TSLA", "GLD", "TLT", "XOM", "JPM", "KO", "PEP", "VNQ"]
period_start = "2022-01-01"
period_end   = "2024-01-01"

results = []
for ticker in universe:
    h = run_hurst_analysis(HurstInput(
        symbol=ticker,
        start_date=period_start,
        end_date=period_end,
        method="dfa",
    ))
    results.append((ticker, h.hurst, h.regime, h.fit_r_squared))

# Sort by Hurst — most trending to most mean-reverting
results.sort(key=lambda x: -x[1])

print(f"{'Ticker':<8} {'H':>6}  {'Regime':<16} {'R²':>6}")
print("-" * 42)
for ticker, hurst, regime, r2 in results:
    flag = ""
    if regime == "trending":          flag = "← trend-follow"
    elif regime == "mean_reverting":  flag = "← mean-revert"
    else:                             flag = "← skip"
    print(f"{ticker:<8} {hurst:>6.3f}  {regime:<16} {r2:>6.3f}  {flag}")

# Bucket the universe for strategy routing
trending      = [t for t, h, r, _ in results if r == "trending"]
mean_reverting = [t for t, h, r, _ in results if r == "mean_reverting"]
random_walk   = [t for t, h, r, _ in results if r == "random_walk"]

print(f"\nTrend-follow pool    : {trending}")
print(f"Mean-reversion pool  : {mean_reverting}")
print(f"No-edge (skip)       : {random_walk}")
```

**Rolling regime tracking — detecting regime shifts over time:**

```python
from standard_quant_tools.agent.tools import run_hurst_analysis
from standard_quant_tools.agent.models import HurstInput

# Run with rolling_window to see how the regime has evolved
result = run_hurst_analysis(HurstInput(
    symbol="SPY",
    start_date="2019-01-01",
    end_date="2024-01-01",
    method="dfa",
    rolling_window=252,      # 1-year rolling window, slides bar by bar
))

print(f"Full-period Hurst : {result.hurst:.4f}  ({result.regime})")
print(f"Most recent roll  : {result.rolling_current:.4f}")
print()

fracs = result.rolling_regime_fractions
print("Rolling regime history:")
print(f"  Trending       {fracs['trending']:.0%}  of windows")
print(f"  Random walk    {fracs['random_walk']:.0%}  of windows")
print(f"  Mean-reverting {fracs['mean_reverting']:.0%}  of windows")

# Interpreting the fractions:
# If trending > 50%: the asset has historically behaved as a trend-follower's market.
# If mean_reverting > 50%: it has spent most time in a mean-reverting regime.
# If no bucket > 40%: regime has been unstable — strategies must be adaptive.

# Alert: if the current rolling regime differs from the historical majority
dominant = max(fracs, key=fracs.get)
current_regime = (
    "trending" if result.rolling_current > 0.55 else
    "mean_reverting" if result.rolling_current < 0.45 else
    "random_walk"
)
if current_regime != dominant:
    print(f"\n⚠  Regime shift detected: historical={dominant}, current={current_regime}")
    print("   Consider re-evaluating strategy allocation.")
else:
    print(f"\n✓ Regime is consistent with historical behaviour ({dominant})")
```

**Strategy selection matrix by Hurst regime:**

```python
# Reference table: map regime → recommended strategy class
REGIME_PLAYBOOK = {
    "trending": {
        "strategies": ["SMA crossover", "MACD crossover", "Parabolic SAR trailing stop"],
        "avoid":      ["RSI mean-reversion", "Bollinger Band reversion"],
        "sizing_note": "Use wider stops (ATR × 2–3) — trends can retrace before continuing",
        "example_params": {"sma_fast": 10, "sma_slow": 50},
    },
    "random_walk": {
        "strategies": ["MACD crossover (short-term)", "Volatility breakout"],
        "avoid":      ["Long-hold trend systems", "Tight-stop mean-reversion"],
        "sizing_note": "No statistical edge — size down 30–50% vs normal or skip entirely",
        "example_params": {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
    },
    "mean_reverting": {
        "strategies": ["RSI oversold/overbought", "Bollinger Band reversion", "Pairs / spread trading"],
        "avoid":      ["Long SMA crossover systems", "Momentum following"],
        "sizing_note": "Use tighter entry z-scores (±1.5–2 σ) and scale in over 2–3 entries",
        "example_params": {"rsi_period": 14, "oversold": 30, "overbought": 70},
    },
}

from standard_quant_tools.agent.tools import run_hurst_analysis
from standard_quant_tools.agent.models import HurstInput

h = run_hurst_analysis(HurstInput(symbol="NVDA", start_date="2022-01-01", end_date="2024-01-01"))
playbook = REGIME_PLAYBOOK[h.regime]

print(f"NVDA  H={h.hurst:.3f}  →  {h.regime.upper()} regime")
print(f"Recommended  : {', '.join(playbook['strategies'])}")
print(f"Avoid        : {', '.join(playbook['avoid'])}")
print(f"Sizing note  : {playbook['sizing_note']}")
print(f"Example params: {playbook['example_params']}")
```

---

## Model Summary

### Input Models

**Backtest tools (3 models, covering 6 tools — `BacktestInput` is shared by `run_sma_backtest`/`run_rsi_backtest`/`run_macd_backtest`/`run_bollinger_backtest`)**

| Model | Required | Optional (with defaults) |
|---|---|---|
| `BacktestInput` | `symbol`, `start_date`, `end_date`, `strategy_type` | `parameters={}`, `initial_capital=10000`, `commission_pct=0.001`, `slippage_pct=0.0005`, `fill_price="close"` |
| `BuyAndHoldInput` | `symbol`, `start_date`, `end_date` | `initial_capital=10000`, `commission_pct=0.001`, `slippage_pct=0.0005`, `fill_price="close"` |
| `CompareStrategiesInput` | `symbol`, `start_date`, `end_date` | `initial_capital=10000`, `commission_pct=0.001`, `slippage_pct=0.0005`, `sort_by="sharpe_ratio"`, `sma/rsi/macd/bollinger_parameters=None`, `fill_price="close"` |

**Analysis tools (8)**

| Model | Required | Optional (with defaults) |
|---|---|---|
| `AnalysisInput` | `symbol` | `benchmark="SPY"`, `period="1y"` |
| `TechnicalInput` | `symbol`, `start_date`, `end_date` | `indicators=["rsi","macd","bollinger","atr"]` |
| `PortfolioInput` | `tickers`, `weights`, `start_date`, `end_date` | `benchmark="SPY"` |
| `ScreenerInput` | `tickers`, `filters` | `start_date`, `end_date`, `sort_by=None`, `ascending=True` |
| `FactorRegressionInput` | `symbol`, `factor_tickers`, `start_date`, `end_date` | `factor_names=None`, `rolling_window=None` |
| `CointegrationInput` | `symbol_a`, `symbol_b`, `start_date`, `end_date` | `zscore_window=30` |
| `PCAInput` | `tickers`, `start_date`, `end_date` | `n_components=3` (must be ≥ 1) |
| `HurstInput` | `symbol`, `start_date`, `end_date` | `method="dfa"` (`"dfa"`/`"rs"`, strictly validated), `rolling_window=None` |

**Advanced tools (8)**

| Model | Required | Optional (with defaults) |
|---|---|---|
| `PortfolioOptimizationInput` | `tickers`, `start_date`, `end_date` | `method="max_sharpe"` (`"max_sharpe"`/`"min_volatility"`/`"target_return"`/`"target_volatility"`/`"risk_parity"`/`"black_litterman"`), `risk_free_rate=0.0`, `target_return=None`, `target_volatility=None`, `allow_short=False`, `max_weight=None`, `risk_budget=None`, `market_weights=None`, `views=None` (`List[BLViewInput]`), `risk_aversion=2.5`, `tau=0.05`, `periods_per_year=252` — see 09_advanced_agent_tools.md and 05_portfolio.md for the per-method requirements |
| `RegimeAdaptiveInput` | `symbol`, `start_date`, `end_date` | `initial_capital=10000`, `commission_pct=0.001`, `slippage_pct=0.0005`, `hurst_method="dfa"` (`"dfa"`/`"rs"`, strictly validated), `sma/rsi/macd/bollinger_param_grid=None`, `n_workers=1` |
| `RegimeAdaptiveWalkForwardInput` | `symbol`, `start_date`, `end_date` | `train_bars=252`, `test_bars=63`, `initial_capital=10000`, `commission_pct=0.001`, `slippage_pct=0.0005`, `hurst_method="dfa"` (`"dfa"`/`"rs"`, strictly validated), `sma/rsi/macd/bollinger_param_grid=None`, `sort_by="sharpe_ratio"`, `fill_price="close"` (`"close"`/`"next_open"`/`"hl2_exploratory"`, applied to each window's OOS leg) |
| `PairScannerInput` | `tickers`, `start_date`, `end_date` | `max_pairs=10`, `min_half_life=5.0`, `max_half_life=126.0`, `p_value_threshold=0.05`, `zscore_window=30` |
| `WalkForwardInput` | `symbol`, `start_date`, `end_date`, `strategy`, `param_grid` | `train_bars=252`, `test_bars=63`, `initial_capital=10000`, `commission_pct=0.001`, `slippage_pct=0.0005`, `sort_by="sharpe_ratio"`, `fill_price="close"` (`"close"`/`"next_open"`/`"hl2_exploratory"`, applied to each window's OOS leg) |
| `RiskAttributionInput` | `tickers`, `weights`, `start_date`, `end_date` | `benchmark="SPY"`, `n_components=3`, `factor_tickers=None`, `factor_names=None` |
| `PositionSizerInput` | `symbol`, `start_date`, `end_date`, `account_equity` | `risk_per_trade_pct=0.01` (must be in (0,1]), `atr_period=14`, `atr_multiplier=2.0`, `win_rate=None`, `avg_win_pct=None`, `avg_loss_pct=None` |
| `PortfolioSimulationInput` | `tickers`, `start_date`, `end_date`, `target_weights` | `signal_type="target_weight"`, `construction_method=None`, `gross_leverage=1.0`, `n_long=None`, `n_short=None`, `vol_lookback=20`, `make_dollar_neutral=False`, `initial_capital=10000`, `commission_pct=0.001`, `slippage_pct=0.0005`, `max_gross_leverage=1.0`, `max_position_pct=1.0`, `fill_price="close"` (`"close"`/`"next_open"`/`"hl2_exploratory"`), `commission_model="pct"`, `per_share_rate=0.0`, `min_commission=0.0`, `use_impact_model=False`, `impact_coefficient=1.0`, `impact_lookback=20`, `borrow_fee_bps=0.0`, `margin_interest_rate=0.0`, `max_adv_participation=None`, `benchmark=None` — see 09_advanced_agent_tools.md for how the cost-model and construction-method fields interact |

**Supplementary tools (6)**

| Model | Required | Optional (with defaults) |
|---|---|---|
| `FundamentalsInput` | `symbol` | — |
| `BacktestOptInput` | `symbol`, `start_date`, `end_date`, `strategy`, `param_grid` | `initial_capital=10000`, `sort_by="sharpe_ratio"`, `top_n=5`, `n_workers=1`, `fill_price="close"` |
| `AdvancedIndicatorsInput` | `symbol`, `start_date`, `end_date` | `mfi_period=14`, `atr_period=14`, `sar_af_start=0.02`, `sar_af_max=0.2` |
| `RollingBetaInput` | `symbol`, `start_date`, `end_date` | `benchmark="SPY"`, `window=60` |
| `ExtendedRiskInput` | `symbol`, `start_date`, `end_date` | `benchmark="SPY"` |
| `BacktestDiagnosticsInput` | `symbol`, `start_date`, `end_date`, `strategy_type` | `parameters={}`, `initial_capital=10000`, `commission_pct=0.001`, `slippage_pct=0.0005`, `top_n_drawdowns=5`, `fill_price="close"` |

**Custom signal tools (2)**

| Model | Required | Optional (with defaults) |
|---|---|---|
| `CustomSignalBacktestInput` | `symbol`, `start_date`, `end_date`, `signals` (`{date: value}`) | `signal_type="direction"`, `max_abs_weight=1.0`, `signal_fill_policy="hold"`, `initial_capital=10000`, `commission_pct=0.001`, `slippage_pct=0.0005`, `fill_price="close"` |
| `SignalPanelBacktestInput` | `tickers`, `start_date`, `end_date`, `signal_panel` (`{ticker: {date: value}}`) | `weights=None` (equal weight), `signal_fill_policy="hold"`, `initial_capital=10000`, `commission_pct=0.001`, `slippage_pct=0.0005`, `benchmark=None`, `include_trade_log=False`, `fill_price="close"`, `signal_type="score"`, `max_abs_weight=1.0` |

**Options tools (2)** — see [12_options.md](12_options.md)

| Model | Required | Optional (with defaults) |
|---|---|---|
| `OptionPricingInput` | `spot`, `strike`, `time_to_expiry`, `risk_free_rate`, `volatility` | `option_type="call"` (`"call"`/`"put"`), `dividend_yield=0.0` |
| `ImpliedVolatilityInput` | `option_price`, `spot`, `strike`, `time_to_expiry`, `risk_free_rate` | `option_type="call"`, `dividend_yield=0.0` |

**`signal_type` — what a custom signal's values mean, opt-in validation:**

| `signal_type` | Meaning | Validation |
|---|---|---|
| `"score"` | Unrestricted — you own the scale/leverage semantics | None — exactly today's original permissive behavior |
| `"direction"` (default for `CustomSignalBacktestInput`) | Position direction | Every value must be exactly `-1`, `0`, or `1` |
| `"target_weight"` | Portfolio weight for this position | Every `\|value\|` must be ≤ `max_abs_weight` (default `1.0`) |

`run_strategy`'s math never changes based on `signal_type` — it always multiplies the (lagged) signal value by the bar's return: `strategy_return = lagged_signal * market_return`. Under `"score"`, that value is a literal leverage multiplier (a value of 10 means a 10x position), not a normalized "confidence score" — which is why `CustomSignalBacktestInput` defaults to `"direction"` rather than `"score"`: an LLM-facing single-asset tool should not silently accept an arbitrary score as if it were a bounded confidence value (an approved breaking change from the prior default). `SignalPanelBacktestInput` and `PortfolioSimulationInput` still default to `"score"`, because in both of those a score is converted into a bounded weight via an explicit `construction_method` (`backtest/sizing.py`) before it ever reaches a return calculation, so the same hazard doesn't apply. `signal_type` only controls whether malformed values are rejected up front with a Pydantic `ValidationError` naming the offending date/ticker/value, instead of silently backtesting a typo. For `SignalPanelBacktestInput`, the chosen mode applies uniformly across every ticker in `signal_panel`, and validation errors name which ticker failed.

**Validation rules (Pydantic v2):**
- `PortfolioInput` and `RiskAttributionInput`: `weights` must sum to 1.0 and `len(weights) == len(tickers)`.
- `PCAInput`: `n_components` must be ≥ 1.
- `PositionSizerInput`: `risk_per_trade_pct` must be in (0, 1].
- `SignalPanelBacktestInput`: `signal_panel` must have an entry for every ticker in `tickers`; if `weights` is given, its keys must exactly match `tickers` and sum to 1.0.
- `CustomSignalBacktestInput` / `SignalPanelBacktestInput`: signal values must satisfy `signal_type`'s constraint (see table above).
- `PortfolioSimulationInput`: `target_weights` must have an entry for every ticker in `tickers`, and every ticker must share the identical set of rebalance dates. When `signal_type="target_weight"` (default), each date's weights must also satisfy the `target_weight` constraint and gross leverage must not exceed `max_gross_leverage`. When `signal_type="score"`, `construction_method` is required (and `n_long`/`n_short` are required when it is `"equal_weight_top_bottom"`).

### Output Models

**Backtest tools**

| Model | Key fields |
|---|---|
| `BacktestResult` | `total_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `win_rate`, `profit_factor`, `num_trades`, `avg_trade_return_pct`, `final_equity`, `equity_curve`, `trade_log` |
| `StrategyComparison` | `strategy`, `parameters`, `total_return`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `win_rate`, `num_trades`, `final_equity` |
| `CompareStrategiesResult` | `symbol`, `sort_by`, `best_strategy`, `buy_and_hold_return`, `strategies` (List[StrategyComparison]) |

**Analysis tools**

| Model | Key fields |
|---|---|
| `AnalysisResult` | `symbol`, `benchmark`, `alpha`, `beta`, `r_squared`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `var_95`, `cvar_95`, `information_ratio` |
| `TechnicalResult` | `symbol`, `last_close`, `last_values` (dict), `signals` (dict) |
| `PortfolioResult` | `tickers`, `weights`, `annualized_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `var_95`, `cvar_95`, `information_ratio`, `total_return`, `correlation_matrix` |
| `ScreenerResult` | `num_passed`, `tickers_passed`, `results` (list of dicts), `failed_filters` (`Dict[str, str]` — ticker → filter key it failed), `failed_tickers` (`Dict[str, str]` — ticker → data-fetch/compute error), `failed_batches` (`List[str]` — worker-batch errors when `n_workers > 1`) |
| `Trade` | `entry_date`, `exit_date`, `direction`, `entry_price` (weighted-average cost basis over the lot — a computed basis, not necessarily a traded level), `exit_price`, `position_size` (the lot's **peak** signed exposure — 1.0/-1.0 for DIRECTION, fractional/leveraged for SCORE, and the largest size held if the lot was resized; `return_pct` scales with it), `return_pct` |
| `FactorRegressionResult` | `symbol`, `factors`, `alpha`, `loadings`, `t_stats`, `p_values`, `r_squared`, `adj_r_squared`, `n_obs`, `rolling_alpha_tail`, `rolling_loadings_tail` |
| `CointegrationResult` | `symbol_a`, `symbol_b`, `cointegrated`, `p_value`, `hedge_ratio`, `adf_statistic`, `half_life_days`, `critical_values`, `spread_mean`, `spread_std`, `current_zscore`, `signal`, `n_obs` |
| `PCAResult` | `tickers`, `n_components`, `n_obs`, `explained_variance_ratio`, `cumulative_variance_ratio`, `loadings`, `factor_contributions` |
| `HurstResult` | `symbol`, `hurst`, `regime`, `fit_r_squared`, `method`, `n_obs`, `rolling_current`, `rolling_regime_fractions` |

**Advanced tools**

| Model | Key fields |
|---|---|
| `RegimeAdaptiveResult` | `symbol`, `regime`, `hurst`, `fit_r_squared`, `selected_strategy`, `best_parameters`, `grid_combinations`, `backtest` (full `BacktestResult`) |
| `RegimeAdaptiveWalkForwardResult` | `symbol`, `n_windows`, `windows` (List[`RegimeAdaptiveWalkForwardWindow`]), `avg_oos_sharpe`, `avg_oos_return`, `avg_oos_max_drawdown`, `pct_windows_profitable`, `strategy_stability`, `stitched_oos_return`, `stitched_oos_sharpe`, `stitched_oos_sortino`, `stitched_oos_max_drawdown`, `stitched_oos_calmar`, `worst_oos_window`, `longest_losing_window_streak` |
| `RegimeAdaptiveWalkForwardWindow` | `window_index`, `train_start`, `train_end`, `test_start`, `test_end`, `regime`, `hurst`, `fit_r_squared`, `selected_strategy`, `best_params`, `in_sample_sharpe`, `in_sample_return`, `out_of_sample_sharpe`, `out_of_sample_return`, `out_of_sample_max_drawdown` |
| `PairScannerResult` | `n_pairs_tested`, `n_pairs_cointegrated`, `n_pairs_returned`, `pairs` (List[`PairResult`]), `failed_pairs` (List[`PairFailure`]), `failed_tickers` (`Dict[str, str]`) |
| `PairResult` | `symbol_a`, `symbol_b`, `p_value`, `hedge_ratio`, `half_life_days`, `adf_statistic`, `current_zscore`, `signal` |
| `PairFailure` | `symbol_a`, `symbol_b`, `reason` |
| `WalkForwardResult` | `symbol`, `strategy`, `n_windows`, `windows` (List[`WalkForwardWindow`]), `avg_oos_sharpe`, `avg_oos_return`, `avg_oos_max_drawdown`, `pct_windows_profitable`, `param_stability`, `stitched_oos_return`, `stitched_oos_sharpe`, `stitched_oos_sortino`, `stitched_oos_max_drawdown`, `stitched_oos_calmar`, `is_to_oos_sharpe_decay`, `is_to_oos_return_decay`, `worst_oos_window`, `longest_losing_window_streak`, `parameter_turnover` |
| `WalkForwardWindow` | `window_index`, `train_start`, `train_end`, `test_start`, `test_end`, `best_params`, `in_sample_sharpe`, `in_sample_return`, `out_of_sample_sharpe`, `out_of_sample_return`, `out_of_sample_max_drawdown` |
| `RiskAttributionResult` | `tickers`, `weights`, `annualized_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `var_95`, `cvar_95`, `information_ratio`, `asset_risk_contributions`, `pca_variance_explained`, `portfolio_pc_exposures`, `factor_loadings`, `factor_r_squared`, `factor_alpha` |
| `PortfolioOptimizationResult` | `tickers`, `method`, `weights`, `expected_return`, `expected_volatility`, `sharpe_ratio`, `converged`, `risk_contributions` (risk_parity only), `warnings` |
| `PositionSizerResult` | `symbol`, `last_close`, `atr`, `atr_pct`, `stop_distance`, `shares_fixed_risk`, `position_value_fixed_risk`, `portfolio_pct_fixed_risk`, `max_loss_fixed_risk`, `kelly_fraction`, `shares_half_kelly`, `position_value_half_kelly`, `portfolio_pct_half_kelly`, `recommended_sizing`, `recommended_shares`, `recommended_position_value` |
| `PortfolioSimulationResult` | `tickers`, `n_rebalances`, `rebalance_log` (List[`RebalanceEvent`]), `total_return`, `annualized_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `var_95`, `cvar_95`, `information_ratio`, `final_equity`, `final_cash`, `avg_gross_leverage`, `max_gross_leverage_used`, `equity_curve`, `warnings` |
| `RebalanceEvent` | `date`, `turnover_pct`, `gross_leverage_after`, `n_positions` |

**Supplementary tools**

| Model | Key fields |
|---|---|
| `FundamentalsResult` | `symbol`, `name`, `sector`, `industry`, `country`, `full_time_employees`, `market_cap`, `trailing_pe`, `forward_pe`, `price_to_book`, `debt_to_equity`, `return_on_equity`, `profit_margins`, `dividend_yield` |
| `OptimizationRun` | `rank`, `parameters`, `total_return`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `num_trades` |
| `BacktestOptResult` | `symbol`, `strategy`, `n_combinations`, `sort_by`, `best_params`, `best_sharpe`, `best_return`, `top_results` (List[`OptimizationRun`]) |
| `AdvancedIndicatorsResult` | `symbol`, `last_close`, `sar_value`, `sar_trend`, `sar_signal`, `wilder_atr`, `wilder_atr_pct`, `mfi`, `mfi_signal` |
| `RollingBetaResult` | `symbol`, `benchmark`, `window`, `current_beta`, `beta_1m_ago`, `beta_3m_ago`, `beta_6m_ago`, `beta_trend`, `beta_min`, `beta_max`, `beta_mean`, `n_obs` |
| `ExtendedRiskResult` | `symbol`, `benchmark`, `annualized_return`, `calmar_ratio`, `treynor_ratio`, `var_parametric_95`, `var_parametric_99`, `var_historical_99`, `cvar_99`, `beta` |
| `BacktestDiagnosticsResult` | `symbol`, `strategy_type`, `total_return`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `num_trades`, `top_drawdowns` (List[`DrawdownEpisode`]), `trade_diagnostics` (`TradeDiagnostics`), `exposure` (`ExposureDiagnostics`) |
| `DrawdownEpisode` | `start`, `trough`, `end`, `depth`, `duration_bars`, `recovery_bars` |
| `TradeDiagnostics` | `expectancy_pct`, `avg_winner_pct`, `avg_loser_pct`, `payoff_ratio`, `max_consecutive_wins`, `max_consecutive_losses`, `avg_mae_pct`, `avg_mfe_pct` |
| `ExposureDiagnostics` | `time_in_market`, `avg_gross_exposure`, `avg_net_exposure`, `pct_long`, `pct_short`, `avg_holding_period_bars` |

**Custom signal tools**

| Model | Key fields |
|---|---|
| `run_custom_signal_backtest` output | Reuses `BacktestResult` — identical shape to the built-in strategy backtests |
| `SignalPanelBacktestResult` | `tickers`, `per_ticker` (`Dict[str, BacktestResult]`), `portfolio_metrics` (same shape as `portfolio.portfolio_metrics()` output: `annualized_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `var_95`, `cvar_95`, `total_return`, `tickers`, `weights`, plus `information_ratio` when `benchmark` was set) |

**Options tools**

| Model | Key fields |
|---|---|
| `OptionPricingResult` | `option_type`, `price`, `greeks` (`OptionGreeks`), `d1`, `d2` |
| `OptionGreeks` | `delta`, `gamma`, `vega` (per 1.0 of volatility, not per vol point), `theta` (per year, not per day), `rho` |
| `ImpliedVolatilityResult` | `implied_volatility`, `converged`, `iterations`, `method` (`"newton"`/`"bisection"`) |

---

## Advanced Tools

The remaining 31 advanced, supplementary, custom-signal, analytics, options, and diagnostic tools compose existing primitives into single, LLM-callable operations covering complete research workflows. Full documentation with output reference tables and multi-step chaining examples is in [Documentation/09_advanced_agent_tools.md](09_advanced_agent_tools.md).
