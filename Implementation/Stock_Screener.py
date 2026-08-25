"""
Stock Screener — filter a large universe down to high-quality candidates
using a combination of technical and fundamental criteria, then run a
quick risk profile on each passing stock.

Demonstrates:
  - run_screener(): multi-filter universe screener
  - analyze_stock_risk(): per-stock beta/alpha/Sharpe/VaR summary
  - get_position_size(): Kelly + ATR sizing for each candidate
"""

import datetime
import logging
from pathlib import Path

from standard_quant_tools.agent.tools import (
    run_screener,
    analyze_stock_risk,
    get_position_size,
)
from standard_quant_tools.agent.models import (
    ScreenerInput,
    AnalysisInput,
    PositionSizerInput,
)

# ── Logging ────────────────────────────────────────────────────────
_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

_fmt = logging.Formatter(
    "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
_fh = logging.FileHandler(_LOGS_DIR / f"stock_screener_{_ts}.log", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("  %(levelname)-7s  %(message)s"))

_lib = logging.getLogger("standard_quant_tools")
_lib.setLevel(logging.DEBUG)
_lib.addHandler(_fh)
_lib.addHandler(_sh)

# ── Configuration ──────────────────────────────────────────────────
# Mid/large-cap US tech + consumer universe
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
    "LYFT",
    "ABNB",
    "PYPL",
    "SQ",
    "INTC",
    "AMD",
]

ACCOUNT = 500_000.0
START_DATE = "2023-01-01"
END_DATE = datetime.date.today().isoformat()

# ── Screen filters ─────────────────────────────────────────────────
# Look for quality growth names that aren't overbought
FILTERS = {
    "rsi_max": 60,  # not overbought (RSI < 60)
    "rsi_min": 30,  # not in free-fall (RSI > 30)
    "price_above_sma": 50,  # price above 50-day SMA (uptrend)
    "beta_max": 1.8,  # not excessively volatile vs market
    "beta_min": 0.5,  # has meaningful market correlation
}

# ── Step 1: Screen universe ────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"  Stock Screener")
print(f"  Universe: {len(UNIVERSE)} stocks  |  Filters: {list(FILTERS.keys())}")
print(f"{'═'*65}\n")

screen = run_screener(
    ScreenerInput(
        tickers=UNIVERSE,
        filters=FILTERS,
        start_date=START_DATE,
        end_date=END_DATE,
        sort_by="rsi_14",
        ascending=True,  # lowest RSI first (most room to run)
    )
)

print(f"  Passed : {screen.num_passed} / {len(UNIVERSE)}\n")

if not screen.tickers_passed:
    print("  No stocks passed the filters. Try relaxing criteria.")
    exit()

# Print screener results table
print(f"  {'Ticker':<8}{'Close':>8}{'RSI':>7}{'SMA50':>9}{'Beta':>7}")
print("  " + "─" * 44)
for row in screen.results:
    ticker = str(row.get("ticker", ""))
    close = row.get("last_close", float("nan"))
    rsi = row.get("rsi_14", float("nan"))
    sma = row.get("sma_50", float("nan"))
    beta = row.get("beta", float("nan"))
    print(
        f"  {ticker:<8}"
        f"${close:>7.2f}"
        f"{rsi:>7.1f}"
        f"${sma:>8.2f}"
        f"{beta:>7.3f}"
    )

# ── Step 2: Risk profile for each passing stock ────────────────────
print(f"\n{'═'*65}")
print(f"  Risk Profile (2-year vs SPY)")
print(f"{'═'*65}")
print(
    f"\n  {'Ticker':<8}{'Beta':>7}{'Alpha(ann)':>11}{'Sharpe':>8}{'VaR95':>8}{'MaxDD':>9}"
)
print("  " + "─" * 55)

passing_risk = {}
for ticker in screen.tickers_passed:
    risk = analyze_stock_risk(
        AnalysisInput(symbol=ticker, benchmark="SPY", period="2y")
    )
    passing_risk[ticker] = risk
    print(
        f"  {ticker:<8}"
        f"{risk.beta:>7.3f}"
        f"{risk.alpha*252*100:>+10.2f}%"
        f"{risk.sharpe_ratio:>8.3f}"
        f"{risk.var_95*100:>7.2f}%"
        f"{risk.max_drawdown*100:>8.1f}%"
    )

# ── Step 3: Position sizing for each candidate ─────────────────────
print(f"\n{'═'*65}")
print(f"  Position Sizing  (1% risk per trade, ATR(14) × 2 stop)")
print(f"{'═'*65}")
print(
    f"\n  {'Ticker':<8}{'Close':>8}{'ATR%':>7}{'Shares':>8}{'Pos$':>10}{'Port%':>7}{'MaxLoss':>9}"
)
print("  " + "─" * 62)

for ticker in screen.tickers_passed:
    risk = passing_risk[ticker]
    pos = get_position_size(
        PositionSizerInput(
            symbol=ticker,
            start_date=START_DATE,
            end_date=END_DATE,
            account_equity=ACCOUNT,
            risk_per_trade_pct=0.01,
            atr_period=14,
            atr_multiplier=2.0,
            # Optional Kelly inputs from risk profile
            win_rate=None,  # set to e.g. risk.win_rate if available from backtest
            avg_win_pct=None,
            avg_loss_pct=None,
        )
    )
    print(
        f"  {ticker:<8}"
        f"${pos.last_close:>7.2f}"
        f"{pos.atr_pct:>6.1f}%"
        f"{pos.recommended_shares:>8}"
        f"${pos.recommended_position_value:>9,.0f}"
        f"{pos.portfolio_pct_fixed_risk*100:>6.1f}%"
        f"${pos.max_loss_fixed_risk:>8,.0f}"
    )

print()
print(f"  Account equity: ${ACCOUNT:,.0f}")
total_deployed = sum(
    get_position_size(
        PositionSizerInput(
            symbol=t,
            start_date=START_DATE,
            end_date=END_DATE,
            account_equity=ACCOUNT,
            risk_per_trade_pct=0.01,
        )
    ).recommended_position_value
    for t in screen.tickers_passed
)
print(
    f"  Total deployed if all entered: ${total_deployed:,.0f} "
    f"({100*total_deployed/ACCOUNT:.1f}% of account)\n"
)
