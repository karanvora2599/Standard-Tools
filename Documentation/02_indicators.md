# Technical Indicators

All indicators accept `pd.Series` (or `pd.DataFrame` for OHLCV-based indicators) and return `pd.Series` or `pd.DataFrame` with the same DatetimeIndex as the input — safe to assign back to any DataFrame.

Performance-critical indicators use a three-tier execution stack: **C++ extension** (`_sqt_core`) → **Numba JIT** → **pure Python fallback**. The C++ path is fastest and has no NumPy-version dependency. Numba requires `numba` installed with NumPy ≤ 2.0 (on NumPy 2.x the JIT is a no-op). All functions remain correct regardless of which tier is active — the selection is transparent to callers.

The following indicators have a **C++ fast path** via `_sqt_core`: RSI, ADX, Parabolic SAR, Wilder's ATR, Bollinger Bands, and Stochastic Oscillator. `atr()` uses a **NumPy single-pass** true range computation (`np.maximum`) that is 5.6× faster than the `pd.concat` approach on a 2 000-bar series. All C++ paths fall back to pure Python/pandas automatically when the extension is not built — the API is identical either way.

Internally, when the AI-agent technical-analysis tool (`get_technical_analysis`) needs 2 or more of {RSI, ADX, Bollinger Bands, Stochastic Oscillator} at once, it uses a single fused native call (`technical_indicators()` in `_sqt_core`) instead of one C++ round trip per indicator — measured ~4.6× faster than calling the individual Python wrappers separately at that integration point (n=2 000; the win is eliminating redundant Python-side validation/logging/conversion overhead per call, not a faster indicator kernel — see `Development/performance_insights.md`). This is purely an internal optimization; the individual `rsi()`/`adx()`/`bollinger_bands()`/`stochastic_oscillator()` functions documented below are unaffected and still used standalone everywhere else.

---

## Input Validation Contract

Every indicator validates its inputs at the Python boundary, before any
execution tier is chosen — so the same call raises the same error whether or
not `_sqt_core` is built.

| Check | Applies to | Behavior |
|---|---|---|
| `period`/`window` > 0 | all periodised indicators | `ValidationError`. `bollinger_bands` requires ≥ 2 (it needs a sample variance); `macd` additionally requires `fast < slow`. |
| Equal input lengths | multi-series indicators (`adx`, `atr`, `wilder_atr`, `williams_r`, `parabolic_sar`, `vwap`, `mfi`) | `ValidationError` naming the actual lengths. |
| Finite (no NaN/Inf) | `rsi`, `adx`, `atr`, `wilder_atr`, `parabolic_sar`, `bollinger_bands`, `stochastic_oscillator` | `ValidationError` reporting how many non-finite values were found. |

Two of these were genuine safety fixes rather than ergonomics, and are worth
knowing about if you call the kernels in unusual ways:

- **Equal-length validation closes an out-of-bounds read.** The Numba kernels
  size their output from one series and index the others positionally.
  `@njit` compiles with bounds checking *disabled*, so a shorter `low` or
  `high` was an out-of-bounds read that returned plausible-looking numbers
  instead of raising.
- **`adx` with `period >= len(close)` now returns all-NaN on every tier.**
  Previously the Numba kernel wrote past the end of its own output array in
  that case (again, no bounds checking under `@njit`) while the C++ kernel
  returned all-NaN correctly and the pure-Python fallback raised
  `IndexError` — three tiers, three behaviors. `parabolic_sar` on an empty
  series had the same shape.

**Degenerate windows.** A window with zero price range (perfectly flat
prices across the whole lookback) makes `%K` and `%R` a `0/0`:

- `williams_r` yields **NaN** — "position within the range" is undefined when
  there is no range.
- `stochastic_oscillator` yields **0.0**, matching the compiled kernel's
  convention. The pandas fallback previously yielded NaN here while C++ gave
  `0.0`, so the answer depended only on whether the extension was built; the
  fallback now follows C++. Warm-up bars remain NaN on both.
- `mfi` yields **NaN** for a window with no money flow at all (both positive
  and negative flow zero, e.g. zero volume) — it previously reported `0.0`,
  i.e. "maximally oversold", for a window carrying no information.

---

## Trend Indicators

### SMA — Simple Moving Average

```python
from standard_quant_tools.indicators import sma

df['SMA_20'] = sma(df['Close'], period=20)
df['SMA_50'] = sma(df['Close'], period=50)
df['SMA_200'] = sma(df['Close'], period=200)

# Golden cross signal
df['golden_cross'] = df['SMA_50'] > df['SMA_200']
```

### EMA — Exponential Moving Average

```python
from standard_quant_tools.indicators import ema

df['EMA_12'] = ema(df['Close'], period=12)
df['EMA_26'] = ema(df['Close'], period=26)
```

EMA weights recent bars more heavily than SMA, so it reacts faster to price changes.

### MACD

```python
from standard_quant_tools.indicators import macd

macd_df = macd(df['Close'], fast=12, slow=26, signal=9)
df['MACD'] = macd_df['MACD']
df['MACD_Signal'] = macd_df['Signal']
df['MACD_Hist'] = macd_df['Histogram']

# Bullish crossover signal
df['macd_cross_up'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
```

### ADX — Average Directional Index *(C++ / Numba JIT)*

ADX measures trend **strength**, not direction. DI+/DI− give direction.

```python
from standard_quant_tools.indicators import adx

adx_df = adx(df['High'], df['Low'], df['Close'], period=14)
df['ADX'] = adx_df['ADX']
df['DI_Plus'] = adx_df['DI_Plus']
df['DI_Minus'] = adx_df['DI_Minus']

# Strong bullish trend: ADX > 25 and DI+ > DI−
strong_bull = (df['ADX'] > 25) & (df['DI_Plus'] > df['DI_Minus'])
```

| ADX Value | Interpretation |
|---|---|
| < 20 | Weak / no trend |
| 20–25 | Emerging trend |
| 25–50 | Strong trend |
| > 50 | Very strong trend |

### Parabolic SAR *(C++ / Numba JIT)*

A dynamic trailing stop that also signals trend direction.

```python
from standard_quant_tools.indicators import parabolic_sar

sar_df = parabolic_sar(df['High'], df['Low'], af_start=0.02, af_step=0.02, af_max=0.2)
df['SAR'] = sar_df['SAR']
df['Trend'] = sar_df['Trend']   # 1 = rising, -1 = falling

# When SAR flips from -1 to 1: bullish signal
df['sar_flip_bull'] = (df['Trend'] == 1) & (df['Trend'].shift(1) == -1)
```

**Validation:** raises `ValidationError` if `af_start`, `af_step`, or `af_max`
isn't finite, if `af_start <= 0`, if `af_step < 0`, if `af_max <= 0`, or if
`af_max < af_start` — a nonsensical AF combination otherwise produced a
meaningless SAR series silently.

### Williams %R

Momentum oscillator ranging from −100 to 0. Below −80 = oversold; above −20 = overbought.

```python
from standard_quant_tools.indicators import williams_r

df['Williams_R'] = williams_r(df['High'], df['Low'], df['Close'], period=14)
df['wr_oversold'] = df['Williams_R'] < -80
```

---

## Momentum Indicators

### RSI — Relative Strength Index *(C++ / Numba JIT)*

```python
from standard_quant_tools.indicators import rsi

df['RSI'] = rsi(df['Close'], period=14)

# Classic thresholds
df['oversold']   = df['RSI'] < 30
df['overbought'] = df['RSI'] > 70
```

### Stochastic Oscillator *(C++ extension)*

Uses a C++ fused sliding min+max pass — measured 2.6× faster than two separate pandas rolling operations (n=2 000; an earlier, unmeasured 5–15× projection appeared in this doc before `_sqt_core` was actually benchmarked — see `Development/performance_insights.md`). Falls back to pandas when the extension is not built.

```python
from standard_quant_tools.indicators import stochastic_oscillator

stoch = stochastic_oscillator(df['High'], df['Low'], df['Close'], k_period=14, d_period=3)
df['Stoch_K'] = stoch['Stoch_K']
df['Stoch_D'] = stoch['Stoch_D']

# Bullish stochastic cross
df['stoch_bull'] = (df['Stoch_K'] > df['Stoch_D']) & (df['Stoch_K'] < 20)
```

**Validation:** raises `ValidationError` if `k_period <= 0` or `d_period <= 0`
(a `d_period <= 0` previously reached the C++ kernel unchecked and caused an
out-of-bounds read rather than a catchable Python error).

---

## Volatility Indicators

### Bollinger Bands *(C++ extension)*

Uses a C++ fused single-pass mean+std computation — one sliding window maintains both `Σx` and `Σx²`, computing mean and variance together rather than running two separate pandas rolling operations — measured 1.6× faster (n=2 000; an earlier, unmeasured 3–8× projection appeared in this doc before `_sqt_core` was actually benchmarked — see `Development/performance_insights.md`). Falls back to pandas when the extension is not built.

```python
from standard_quant_tools.indicators import bollinger_bands

bb = bollinger_bands(df['Close'], period=20, num_std=2.0)
df['BB_Upper']  = bb['BB_Upper']
df['BB_Middle'] = bb['BB_Middle']   # = SMA(20)
df['BB_Lower']  = bb['BB_Lower']

# Band squeeze: low volatility precedes breakout
df['bb_width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
df['squeeze']  = df['bb_width'] < df['bb_width'].rolling(120).quantile(0.2)
```

### ATR — Average True Range (SMA-smoothed)

ATR measures volatility including overnight gaps. True range is computed in a single `np.maximum` pass (5.6× faster than the `pd.concat` + `.max` approach), then smoothed with a simple rolling mean.

```python
from standard_quant_tools.indicators import atr

df['ATR'] = atr(df['High'], df['Low'], df['Close'], period=14)

# Dynamic stop-loss: 2× ATR below entry
entry_price = float(df['Close'].iloc[-1])
stop_loss = entry_price - 2 * float(df['ATR'].iloc[-1])
```

### Wilder's ATR *(C++ extension)*

`wilder_atr` uses Wilder's exponential smoothing instead of a simple rolling mean — the same recurrence as RSI and ADX:

```
TR[0]   = high[0] − low[0]
TR[i]   = max(H−L, |H−C_prev|, |L−C_prev|)
ATR[p−1] = mean(TR[0..p−1])          ← SMA seed
ATR[i]   = (ATR[i−1]×(p−1) + TR[i]) / p   ← Wilder's smooth
```

This matches the ATR used internally by most charting platforms (TradingView, Bloomberg, MetaTrader) and by the ADX calculation already in this library.

```python
from standard_quant_tools.indicators import wilder_atr

df['Wilder_ATR'] = wilder_atr(df['High'], df['Low'], df['Close'], period=14)

# Position sizing: risk 1% of equity, stop = 2× ATR
account_equity = 100_000
risk_per_trade = account_equity * 0.01
stop_distance  = 2 * float(df['Wilder_ATR'].iloc[-1])
shares         = int(risk_per_trade / stop_distance)

# Volatility filter: avoid trading when ATR is unusually high
atr_pct = df['Wilder_ATR'] / df['Close']          # ATR as % of price
df['low_vol']  = atr_pct < atr_pct.rolling(60).quantile(0.40)
df['high_vol'] = atr_pct > atr_pct.rolling(60).quantile(0.80)
```

**Difference from `atr()`:**

| | `atr()` | `wilder_atr()` |
|---|---|---|
| Smoothing | Simple rolling mean | Wilder's exponential (SMA seed) |
| NaN prefix | First `period` values | First `period−1` values |
| Implementation | NumPy + pandas | **C++ extension** / Python fallback |
| Matches charting platforms | Sometimes | Yes (TradingView, Bloomberg) |

**Check which path is active:**

```python
from standard_quant_tools.indicators.volatility import HAS_CPP
print("C++ Wilder ATR active:", HAS_CPP)
```

---

## Volume Indicators

### OBV — On Balance Volume

```python
from standard_quant_tools.indicators import obv

df['OBV'] = obv(df['Close'], df['Volume'])

# OBV rising while price flat → accumulation
df['obv_divergence'] = (df['OBV'].diff(5) > 0) & (df['Close'].diff(5) < 0)
```

### VWAP — Volume Weighted Average Price

```python
from standard_quant_tools.indicators import vwap

# Cumulative (session) VWAP
df['VWAP'] = vwap(df['High'], df['Low'], df['Close'], df['Volume'])

# Rolling 20-day VWAP
df['VWAP_20'] = vwap(df['High'], df['Low'], df['Close'], df['Volume'], period=20)

# Price above VWAP → bullish intraday bias
df['above_vwap'] = df['Close'] > df['VWAP']
```

### MFI — Money Flow Index

Volume-weighted RSI. Values below 20 = oversold; above 80 = overbought.

```python
from standard_quant_tools.indicators import mfi

df['MFI'] = mfi(df['High'], df['Low'], df['Close'], df['Volume'], period=14)
df['mfi_oversold']   = df['MFI'] < 20
df['mfi_overbought'] = df['MFI'] > 80
```

---

## Whole-Universe (Panel) Computation *(C++ extension)*

Every indicator above takes one `Series` and returns one `Series`/`DataFrame`. That is the
right shape for a single ticker and the wrong shape for a universe: measured at 2 000
tickers × 2 000 bars, the per-ticker call costs **318 µs of Python wrapper against 19 µs of
actual kernel**.

The C++ call boundary is not the problem — pybind11 dispatch is 2.7 µs, 14% of the
per-ticker cost. The pandas round trip is: `Series` → NumPy, validation, logging, and
`Series` reconstruction. So `indicators.panel` converts the universe once, hands the
native side one matrix, and gets one matrix back, with tickers computed in parallel.

```python
from standard_quant_tools.indicators.panel import technical_indicators_panel

# ohlcv_by_ticker: dict of ticker -> OHLCV DataFrame (needs High/Low/Close)
out = technical_indicators_panel(
    ohlcv_by_ticker,
    ["rsi", "adx", "atr", "bollinger_bands", "stochastic_oscillator"],
)

out["rsi"]                      # (dates × tickers)
out["bollinger_bands"]["AAPL"]  # (dates × [BB_Upper, BB_Middle, BB_Lower])
```

Single-valued indicators (`rsi`, `atr`) come back as one column per ticker. Multi-column
ones (`adx`, `bollinger_bands`, `stochastic_oscillator`) use a `(ticker, field)` MultiIndex
on the columns, with **the same field names the per-ticker functions use** — `BB_Upper`,
`DI_Plus`, `Stoch_K` — so there is no second vocabulary to learn.

**Measured**, 500 tickers × 1 000 bars, all five indicators: **144.7 ms**, against 1 727.6
ms for the per-ticker wrapper loop (11.9×).

Parameters (`rsi_period`, `bollinger_num_std`, …) are keyword-only and apply to every
ticker. Arithmetic is identical: each row is handed to the same kernel the single-series
path uses, so output is bit-identical to calling the per-ticker function in a loop.

**The index is the intersection** of every ticker's bars — the only shape a dense panel can
have. A ticker with a shorter history therefore truncates the panel for everyone, which is
a real difference from computing each ticker on its own full history.

---

## Multi-Indicator Example: Full Technical Dashboard

```python
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.indicators import (
    sma, ema, macd, rsi, bollinger_bands, atr, wilder_atr,
    adx, parabolic_sar, obv, vwap, mfi
)

provider = DataFactory.get_provider()
df = provider.get_ohlcv("NVDA", "2023-01-01", "2024-01-01")

# Trend
df['SMA_50']  = sma(df['Close'], 50)
df['EMA_12']  = ema(df['Close'], 12)
macd_df = macd(df['Close'])
df = df.join(macd_df)
adx_df = adx(df['High'], df['Low'], df['Close'])
df = df.join(adx_df)
df = df.join(parabolic_sar(df['High'], df['Low']))

# Momentum & Volatility
df['RSI']         = rsi(df['Close'], 14)
bb = bollinger_bands(df['Close'])
df = df.join(bb)
df['ATR']         = atr(df['High'], df['Low'], df['Close'])
df['Wilder_ATR']  = wilder_atr(df['High'], df['Low'], df['Close'])

# Volume
df['OBV']  = obv(df['Close'], df['Volume'])
df['VWAP'] = vwap(df['High'], df['Low'], df['Close'], df['Volume'])
df['MFI']  = mfi(df['High'], df['Low'], df['Close'], df['Volume'])

print(df.tail())
```
