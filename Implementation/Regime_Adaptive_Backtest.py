"""
Regime-Adaptive Backtest — automatically detect the market regime via the
Hurst exponent, then select and optimise the best-fit strategy via grid search.

Pipeline:
  1. Compute Hurst H on the full price series
     H > 0.55  →  trending      → SMA crossover
     H < 0.45  →  mean-reverting → RSI mean-reversion
     otherwise  →  random walk   → MACD crossover
  2. Grid-search the chosen strategy's parameters on Sharpe ratio
  3. Run the final backtest with the best parameters
  4. Optionally run walk-forward validation to check out-of-sample robustness
"""

import datetime
import logging
from pathlib import Path

from standard_quant_tools.agent.tools import (
    run_regime_adaptive_backtest,
    run_hurst_analysis,
    run_walk_forward_backtest,
)
from standard_quant_tools.agent.models import (
    RegimeAdaptiveInput,
    HurstInput,
    WalkForwardInput,
)

# ── Logging ────────────────────────────────────────────────────────
_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

_fmt = logging.Formatter("%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)s  %(message)s",
                         datefmt="%H:%M:%S")
_fh = logging.FileHandler(_LOGS_DIR / f"regime_adaptive_{_ts}.log", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("  %(levelname)-7s  %(message)s"))

_lib = logging.getLogger("standard_quant_tools")
_lib.setLevel(logging.DEBUG)
_lib.addHandler(_fh)
_lib.addHandler(_sh)

# ── Configuration ──────────────────────────────────────────────────
SYMBOL     = "QQQ"
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"
CAPITAL    = 100_000.0

# ── Step 1: Hurst regime snapshot ─────────────────────────────────
print(f"\n{'═'*65}")
print(f"  Regime Detection — {SYMBOL}  ({START_DATE} → {END_DATE})")
print(f"{'═'*65}\n")

hurst = run_hurst_analysis(
    HurstInput(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        method="dfa",
        rolling_window=63,   # ~1 quarter
    )
)

print(f"  Hurst exponent  : {hurst.hurst:.4f}  (fit R² = {hurst.fit_r_squared:.4f})")
print(f"  Regime          : {hurst.regime.upper()}")
print(f"  Method          : {hurst.method.upper()}")
print(f"  Observations    : {hurst.n_obs}")

if hurst.rolling_current is not None:
    print(f"\n  Rolling Hurst (last 63-bar window): {hurst.rolling_current:.4f}")
if hurst.rolling_regime_fractions:
    fracs = hurst.rolling_regime_fractions
    print(f"  Trending        : {fracs['trending']*100:.0f}%")
    print(f"  Random walk     : {fracs['random_walk']*100:.0f}%")
    print(f"  Mean-reverting  : {fracs['mean_reverting']*100:.0f}%")

# ── Step 2: Regime-adaptive grid search & backtest ─────────────────
print(f"\n{'═'*65}")
print(f"  Regime-Adaptive Backtest")
print(f"{'═'*65}\n")

result = run_regime_adaptive_backtest(
    RegimeAdaptiveInput(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=CAPITAL,
        hurst_method="dfa",
        n_workers=1,
    )
)

bt = result.backtest
print(f"  Detected regime    : {result.regime.upper()}")
print(f"  Selected strategy  : {result.selected_strategy}")
print(f"  Grid combinations  : {result.grid_combinations}")
print(f"  Best parameters    : {result.best_parameters}")
print()
print(f"  Total return       : {bt.total_return*100:>8.1f}%")
print(f"  Sharpe ratio       : {bt.sharpe_ratio:>8.3f}")
print(f"  Sortino ratio      : {bt.sortino_ratio:>8.3f}")
print(f"  Max drawdown       : {bt.max_drawdown*100:>8.1f}%")
print(f"  Calmar ratio       : {bt.calmar_ratio:>8.3f}")
print(f"  Win rate           : {bt.win_rate*100:>7.0f}%")
print(f"  Trades             : {bt.num_trades}")
print(f"  Final equity       : ${bt.final_equity:,.2f}")

# ── Step 3: Walk-forward validation ────────────────────────────────
print(f"\n{'═'*65}")
print(f"  Walk-Forward Validation  (train=252 bars, test=63 bars)")
print(f"{'═'*65}\n")

# Build the param grid from the best parameters found above
strategy = result.selected_strategy
best_p   = result.best_parameters

# Expand each best param into a 3-value grid centred on it
def _small_grid(val, step, n=3):
    if isinstance(val, int):
        return [max(1, val + step * (i - n // 2)) for i in range(n)]
    return [round(val + step * (i - n // 2), 1) for i in range(n)]

if strategy == "sma_crossover":
    param_grid = {
        "fast_period": _small_grid(best_p["fast_period"], 5),
        "slow_period": _small_grid(best_p["slow_period"], 10),
    }
elif strategy == "rsi_mean_reversion":
    param_grid = {
        "period":      _small_grid(best_p["period"], 3),
        "oversold":    _small_grid(best_p["oversold"], 5),
        "overbought":  _small_grid(best_p["overbought"], 5),
    }
elif strategy == "macd_crossover":
    param_grid = {
        "fast":   _small_grid(best_p["fast"], 2),
        "slow":   _small_grid(best_p["slow"], 3),
        "signal": _small_grid(best_p["signal"], 1),
    }
else:  # bollinger_reversion
    param_grid = {
        "period":  _small_grid(best_p["period"], 5),
        "num_std": _small_grid(best_p.get("num_std", 2.0), 0.25),
    }

wf = run_walk_forward_backtest(
    WalkForwardInput(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        strategy=strategy,
        param_grid=param_grid,
        train_bars=252,
        test_bars=63,
        initial_capital=CAPITAL,
    )
)

print(f"  Windows            : {wf.n_windows}")
print(f"  Avg OOS Sharpe     : {wf.avg_oos_sharpe:>8.3f}")
print(f"  Avg OOS return     : {wf.avg_oos_return*100:>7.1f}%")
print(f"  Avg OOS max DD     : {wf.avg_oos_max_drawdown*100:>7.1f}%")
print(f"  Profitable windows : {wf.pct_windows_profitable*100:>5.0f}%")

print("\n  ── Per-Window Results ─────────────────────────────────")
print(f"  {'Win':<5}{'Period':<26}{'IS Sharpe':>10}{'OOS Sharpe':>11}{'OOS Ret':>9}{'Params'}")
print("  " + "─" * 80)
for w in wf.windows:
    period = f"{w.test_start} → {w.test_end}"
    print(
        f"  {w.window_index:<5}"
        f"{period:<26}"
        f"{w.in_sample_sharpe:>10.3f}"
        f"{w.out_of_sample_sharpe:>11.3f}"
        f"{w.out_of_sample_return*100:>8.1f}%"
        f"  {w.best_params}"
    )

print("\n  ── Parameter Stability ─────────────────────────────────")
for param, info in wf.param_stability.items():
    print(f"  {param:<20}  most common = {info['most_common']}  "
          f"({info['frequency']*100:.0f}% of windows)")
print()
