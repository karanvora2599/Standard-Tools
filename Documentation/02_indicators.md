# Technical Indicators

All indicators accept `pd.Series` (or `pd.DataFrame` for OHLCV-based indicators) and return `pd.Series` or `pd.DataFrame` with the same DatetimeIndex as the input — safe to assign back to any DataFrame.

Performance-critical indicators use a three-tier execution stack: **C++ extension** (`_sqt_core`) → **Numba JIT** → **pure Python fallback**. The C++ path is fastest and has no NumPy-version dependency. Numba requires `numba` installed with NumPy ≤ 2.0 (on NumPy 2.x the JIT is a no-op). All functions remain correct regardless of which tier is active — the selection is transparent to callers.

ATR uses a **NumPy single-pass** true range computation (`np.maximum`) that is 5.6× faster than the `pd.concat` approach on a 2 000-bar series.

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

### Stochastic Oscillator

```python
from standard_quant_tools.indicators import stochastic_oscillator

stoch = stochastic_oscillator(df['High'], df['Low'], df['Close'], k_period=14, d_period=3)
df['Stoch_K'] = stoch['Stoch_K']
df['Stoch_D'] = stoch['Stoch_D']

# Bullish stochastic cross
df['stoch_bull'] = (df['Stoch_K'] > df['Stoch_D']) & (df['Stoch_K'] < 20)
```

---

## Volatility Indicators

### Bollinger Bands

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

### ATR — Average True Range

ATR measures volatility including overnight gaps. True range is computed in a single `np.maximum` pass (5.6× faster than the `pd.concat` + `.max` approach), then smoothed with a rolling mean.

```python
from standard_quant_tools.indicators import atr

df['ATR'] = atr(df['High'], df['Low'], df['Close'], period=14)

# Dynamic stop-loss: 2× ATR below entry
entry_price = float(df['Close'].iloc[-1])
stop_loss = entry_price - 2 * float(df['ATR'].iloc[-1])
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

## Multi-Indicator Example: Full Technical Dashboard

```python
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.indicators import (
    sma, ema, macd, rsi, bollinger_bands, atr,
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
df['RSI']  = rsi(df['Close'], 14)
bb = bollinger_bands(df['Close'])
df = df.join(bb)
df['ATR']  = atr(df['High'], df['Low'], df['Close'])

# Volume
df['OBV']  = obv(df['Close'], df['Volume'])
df['VWAP'] = vwap(df['High'], df['Low'], df['Close'], df['Volume'])
df['MFI']  = mfi(df['High'], df['Low'], df['Close'], df['Volume'])

print(df.tail())
```
