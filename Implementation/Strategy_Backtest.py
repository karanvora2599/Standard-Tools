"""
Strategy Backtest — compare all four built-in strategies on a single ticker.

Runs SMA crossover, RSI mean-reversion, MACD crossover, and Bollinger-band
reversion side-by-side against buy-and-hold, then prints a ranked summary table.
"""

import datetime
import logging
from pathlib import Path

from standard_quant_tools.agent.tools import compare_strategies
from standard_quant_tools.agent.models import CompareStrategiesInput

# ── Logging ────────────────────────────────────────────────────────
_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

_fmt = logging.Formatter("%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)s  %(message)s",
                         datefmt="%H:%M:%S")
_fh = logging.FileHandler(_LOGS_DIR / f"strategy_backtest_{_ts}.log", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("  %(levelname)-7s  %(message)s"))

_lib = logging.getLogger("standard_quant_tools")
_lib.setLevel(logging.DEBUG)
_lib.addHandler(_fh)
_lib.addHandler(_sh)

# ── Configuration ──────────────────────────────────────────────────
SYMBOL     = "AAPL"
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"
CAPITAL    = 100_000.0
SORT_BY    = "sharpe_ratio"   # also try "total_return" or "calmar_ratio"

# ── Run ────────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"  Strategy Comparison — {SYMBOL}  ({START_DATE} → {END_DATE})")
print(f"{'═'*65}\n")

result = compare_strategies(
    CompareStrategiesInput(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=CAPITAL,
        sort_by=SORT_BY,
    )
)

# ── Ranked table ───────────────────────────────────────────────────
header = f"{'Rank':<5}{'Strategy':<25}{'Return':>9}{'Sharpe':>8}{'MaxDD':>9}{'Trades':>8}{'Win%':>7}"
print(header)
print("─" * len(header))

for rank, strat in enumerate(result.strategies, 1):
    print(
        f"{rank:<5}{strat.strategy:<25}"
        f"{strat.total_return*100:>8.1f}%"
        f"{strat.sharpe_ratio:>8.3f}"
        f"{strat.max_drawdown*100:>8.1f}%"
        f"{strat.num_trades:>8}"
        f"{strat.win_rate*100:>6.0f}%"
    )

print()
print(f"  Buy & Hold total return: {result.buy_and_hold_return*100:.1f}%")
print(f"  Winner → {result.best_strategy}\n")
