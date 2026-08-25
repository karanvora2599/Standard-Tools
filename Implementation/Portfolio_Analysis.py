"""
Portfolio Analysis — compute risk-adjusted metrics and factor attribution
for a multi-asset portfolio, then print a detailed breakdown.

Demonstrates:
  - get_portfolio_analysis():  Sharpe, Sortino, VaR, CVaR, correlation matrix
  - get_portfolio_risk_attribution(): marginal risk per asset, PCA decomposition,
    factor loadings (market / size / value via SPY / IWM / IWD proxies)
  - analyze_stock_risk(): per-asset beta/alpha vs SPY
"""

import datetime
import logging
from pathlib import Path

from standard_quant_tools.agent.tools import (
    get_portfolio_analysis,
    get_portfolio_risk_attribution,
    analyze_stock_risk,
)
from standard_quant_tools.agent.models import (
    PortfolioInput,
    RiskAttributionInput,
    AnalysisInput,
)

# ── Logging ────────────────────────────────────────────────────────
_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

_fmt = logging.Formatter(
    "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
_fh = logging.FileHandler(_LOGS_DIR / f"portfolio_analysis_{_ts}.log", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("  %(levelname)-7s  %(message)s"))

_lib = logging.getLogger("standard_quant_tools")
_lib.setLevel(logging.DEBUG)
_lib.addHandler(_fh)
_lib.addHandler(_sh)

# ── Configuration ──────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
WEIGHTS = [0.20, 0.20, 0.20, 0.20, 0.20]  # equal-weight
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
BENCHMARK = "SPY"

# Factor proxies: market (SPY), small-cap (IWM), value (IWD)
FACTOR_TICKERS = ["SPY", "IWM", "IWD"]
FACTOR_NAMES = ["market", "size", "value"]

# ── Portfolio-level metrics ─────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"  Portfolio Metrics — {' / '.join(TICKERS)}")
print(f"  Equal-weight  ({START_DATE} → {END_DATE})")
print(f"{'═'*65}\n")

port = get_portfolio_analysis(
    PortfolioInput(
        tickers=TICKERS,
        weights=WEIGHTS,
        start_date=START_DATE,
        end_date=END_DATE,
        benchmark=BENCHMARK,
    )
)

print(f"  Total return         : {port.total_return*100:>8.1f}%")
print(f"  Annualized return    : {port.annualized_return*100:>8.1f}%")
print(f"  Annualized vol       : {port.annualized_volatility*100:>8.1f}%")
print(f"  Sharpe ratio         : {port.sharpe_ratio:>8.3f}")
print(f"  Sortino ratio        : {port.sortino_ratio:>8.3f}")
print(f"  Calmar ratio         : {port.calmar_ratio:>8.3f}")
print(f"  Max drawdown         : {port.max_drawdown*100:>8.1f}%")
print(f"  VaR 95               : {port.var_95*100:>8.2f}%")
print(f"  CVaR 95              : {port.cvar_95*100:>8.2f}%")
print(f"  Information ratio    : {port.information_ratio:>8.3f}")

print("\n  ── Pairwise Correlation ─────────────────────────────")
corr = port.correlation_matrix
print(f"  {'':12}", end="")
for t in TICKERS:
    print(f"  {t:>6}", end="")
print()
for row_t in TICKERS:
    print(f"  {row_t:<12}", end="")
    for col_t in TICKERS:
        val = corr.get(row_t, {}).get(col_t, float("nan"))
        print(f"  {val:>6.3f}", end="")
    print()

# ── Risk attribution & PCA ─────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"  Risk Attribution & PCA Decomposition")
print(f"{'═'*65}\n")

attr = get_portfolio_risk_attribution(
    RiskAttributionInput(
        tickers=TICKERS,
        weights=WEIGHTS,
        start_date=START_DATE,
        end_date=END_DATE,
        benchmark=BENCHMARK,
        n_components=3,
        factor_tickers=FACTOR_TICKERS,
        factor_names=FACTOR_NAMES,
    )
)

print("  ── Marginal Risk Contribution (fraction of total portfolio variance) ──")
for ticker, frac in attr.asset_risk_contributions.items():
    bar = "█" * int(frac * 40)
    print(f"  {ticker:<6}  {frac*100:5.1f}%  {bar}")

print("\n  ── PCA Variance Explained ──────────────────────────────")
for pc, evr in attr.pca_variance_explained.items():
    bar = "█" * int(evr * 50)
    print(f"  {pc}  {evr*100:5.1f}%  {bar}")

print("\n  ── Portfolio PC Exposures ──────────────────────────────")
for pc, exp in attr.portfolio_pc_exposures.items():
    print(f"  {pc}  {exp:+.4f}")

if attr.factor_loadings:
    print("\n  ── Factor Loadings ─────────────────────────────────────")
    for factor, loading in attr.factor_loadings.items():
        print(f"  {factor:<10}  {loading:+.4f}")
    print(f"  Alpha             {attr.factor_alpha:+.6f}")
    print(f"  R²                {attr.factor_r_squared:.4f}")

# ── Per-asset risk profile ─────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"  Per-Asset Risk vs {BENCHMARK}")
print(f"{'═'*65}")
print(f"\n  {'Ticker':<8}{'Beta':>7}{'Alpha':>9}{'Sharpe':>8}{'VaR95':>8}{'MaxDD':>9}")
print("  " + "─" * 50)

for ticker in TICKERS:
    risk = analyze_stock_risk(
        AnalysisInput(symbol=ticker, benchmark=BENCHMARK, period="2y")
    )
    print(
        f"  {ticker:<8}"
        f"{risk.beta:>7.3f}"
        f"{risk.alpha*252*100:>+8.2f}%"
        f"{risk.sharpe_ratio:>8.3f}"
        f"{risk.var_95*100:>7.2f}%"
        f"{risk.max_drawdown*100:>8.1f}%"
    )

print()
