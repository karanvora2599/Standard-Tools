"""
Pair Trading — scan a universe for cointegrated pairs, rank them by
mean-reversion speed (half-life), and inspect the best pair in detail.

Pipeline:
  1. scan_pairs(): test all O(n²/2) ticker combinations for cointegration
  2. Print a ranked table of cointegrated pairs with current z-score signals
  3. run_cointegration_test(): deep-dive on the top pair with spread stats
  4. get_position_size(): size the trade using ATR-based risk management
"""

import datetime
import logging
from pathlib import Path

from standard_quant_tools.agent.tools import (
    scan_pairs,
    run_cointegration_test,
    get_position_size,
)
from standard_quant_tools.agent.models import (
    PairScannerInput,
    CointegrationInput,
    PositionSizerInput,
)

# ── Logging ────────────────────────────────────────────────────────
_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

_fmt = logging.Formatter("%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)s  %(message)s",
                         datefmt="%H:%M:%S")
_fh = logging.FileHandler(_LOGS_DIR / f"pair_trading_{_ts}.log", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("  %(levelname)-7s  %(message)s"))

_lib = logging.getLogger("standard_quant_tools")
_lib.setLevel(logging.DEBUG)
_lib.addHandler(_fh)
_lib.addHandler(_sh)

# ── Configuration ──────────────────────────────────────────────────
# Energy sector — historically well-cointegrated
UNIVERSE   = ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL"]
START_DATE = "2021-01-01"
END_DATE   = "2024-12-31"
ACCOUNT    = 250_000.0

# ── Step 1: Scan all pairs ─────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"  Cointegration Pair Scanner")
print(f"  Universe: {', '.join(UNIVERSE)}")
print(f"  {START_DATE} → {END_DATE}")
print(f"{'═'*65}\n")

scan = scan_pairs(
    PairScannerInput(
        tickers=UNIVERSE,
        start_date=START_DATE,
        end_date=END_DATE,
        max_pairs=10,
        min_half_life=5.0,
        max_half_life=126.0,   # ~6 months
        p_value_threshold=0.05,
        zscore_window=30,
    )
)

print(f"  Pairs tested       : {scan.n_pairs_tested}")
print(f"  Cointegrated       : {scan.n_pairs_cointegrated} "
      f"({100*scan.n_pairs_cointegrated/max(scan.n_pairs_tested,1):.0f}%)")
print(f"  Returned (top)     : {scan.n_pairs_returned}\n")

if not scan.pairs:
    print("  No cointegrated pairs found. Try widening the half-life range or lowering p-value threshold.")
    exit()

print(f"  {'Rank':<5}{'Pair':<14}{'p-val':>7}{'Hedge':>7}{'Half-Life':>10}{'Z-score':>9}{'Signal'}")
print("  " + "─" * 65)
for i, pair in enumerate(scan.pairs, 1):
    print(
        f"  {i:<5}"
        f"{pair.symbol_a}/{pair.symbol_b:<10}"
        f"{pair.p_value:>7.4f}"
        f"{pair.hedge_ratio:>7.3f}"
        f"{pair.half_life_days:>9.1f}d"
        f"{pair.current_zscore:>9.2f}"
        f"  {pair.signal}"
    )

# ── Step 2: Deep-dive on best pair ────────────────────────────────
best = scan.pairs[0]
print(f"\n{'═'*65}")
print(f"  Deep Dive — {best.symbol_a} / {best.symbol_b}")
print(f"{'═'*65}\n")

detail = run_cointegration_test(
    CointegrationInput(
        symbol_a=best.symbol_a,
        symbol_b=best.symbol_b,
        start_date=START_DATE,
        end_date=END_DATE,
        zscore_window=30,
    )
)

print(f"  Cointegrated       : {detail.cointegrated}")
print(f"  ADF statistic      : {detail.adf_statistic:.4f}")
print(f"  p-value            : {detail.p_value:.4f}")
print(f"  Critical values    : {detail.critical_values}")
print(f"  Hedge ratio        : {detail.hedge_ratio:.4f}  "
      f"(long 1 {detail.symbol_a}, short {detail.hedge_ratio:.3f} {detail.symbol_b})")
print(f"  Half-life          : {detail.half_life_days:.1f} bars")
print(f"  Spread mean        : {detail.spread_mean:.6f}")
print(f"  Spread std         : {detail.spread_std:.6f}")
print(f"  Current z-score    : {detail.current_zscore:.4f}")
print(f"  Signal             : {detail.signal.upper()}")
print(f"  Observations       : {detail.n_obs}")

# ── Step 3: Position sizing ────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"  Position Sizing — {best.symbol_a}  (ATR-based, 1% risk)")
print(f"{'═'*65}\n")

pos = get_position_size(
    PositionSizerInput(
        symbol=best.symbol_a,
        start_date=START_DATE,
        end_date=END_DATE,
        account_equity=ACCOUNT,
        risk_per_trade_pct=0.01,   # risk 1% of account per trade
        atr_period=14,
        atr_multiplier=2.0,
    )
)

print(f"  Last close         : ${pos.last_close:.4f}")
print(f"  ATR(14)            : ${pos.atr:.4f}  ({pos.atr_pct:.2f}% of price)")
print(f"  Stop distance      : ${pos.stop_distance:.4f}  (2× ATR)")
print()
print(f"  Fixed-risk sizing  :")
print(f"    Shares           : {pos.shares_fixed_risk}")
print(f"    Position value   : ${pos.position_value_fixed_risk:,.2f}")
print(f"    Portfolio %      : {pos.portfolio_pct_fixed_risk*100:.1f}%")
print(f"    Max $ loss       : ${pos.max_loss_fixed_risk:,.2f}")
print()
print(f"  Recommended        : {pos.recommended_sizing}  —  "
      f"{pos.recommended_shares} shares  (${pos.recommended_position_value:,.2f})")

if detail.signal != "neutral":
    leg_a = "LONG " if "long_a" in detail.signal else "SHORT"
    leg_b = "SHORT" if "long_a" in detail.signal else "LONG "
    print(f"\n  Trade suggestion:")
    print(f"    {leg_a} {pos.recommended_shares} shares of {detail.symbol_a}")
    print(f"    {leg_b} {int(pos.recommended_shares * detail.hedge_ratio)} shares of {detail.symbol_b}")
    print(f"    Enter when z-score reverts toward 0 (current: {detail.current_zscore:.2f})")

print()
