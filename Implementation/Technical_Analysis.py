"""
Technical Analysis — fetch all available indicators for a ticker and
print a snapshot of the latest bar's values with directional signals.
"""

import datetime
import logging
from pathlib import Path

from standard_quant_tools.agent.tools import get_technical_analysis
from standard_quant_tools.agent.models import TechnicalInput

# ── Logging ────────────────────────────────────────────────────────
_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

_fmt = logging.Formatter(
    "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
_fh = logging.FileHandler(_LOGS_DIR / f"technical_analysis_{_ts}.log", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("  %(levelname)-7s  %(message)s"))

_lib = logging.getLogger("standard_quant_tools")
_lib.setLevel(logging.DEBUG)
_lib.addHandler(_fh)
_lib.addHandler(_sh)

# ── Configuration ──────────────────────────────────────────────────
SYMBOL = "TSLA"
START_DATE = "2023-01-01"
END_DATE = datetime.date.today().isoformat()

# All available indicator groups
INDICATORS = [
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger",
    "atr",
    "obv",
    "vwap",
    "stochastic",
    "adx",
    "williams_r",
]

# ── Run ────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  Technical Analysis — {SYMBOL}  (as of {END_DATE})")
print(f"{'═'*60}\n")

result = get_technical_analysis(
    TechnicalInput(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        indicators=INDICATORS,
    )
)

print(f"  Last close: ${result.last_close:.4f}\n")

print("  ── Indicator Values ──────────────────────────────")
for name, value in sorted(result.last_values.items()):
    print(f"    {name:<28} {value}")

print("\n  ── Active Signals ────────────────────────────────")
active = sorted(k for k, v in result.signals.items() if v)
inactive = sorted(k for k, v in result.signals.items() if not v)

for name in active:
    print(f"    [ON]  {name}")
for name in inactive:
    print(f"    [off] {name}")

print()
