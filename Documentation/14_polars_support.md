# Polars Support (optional)

Standard Quant Tools is pandas-first — pandas stays the default and required
backend everywhere. [Polars](https://pola.rs) support is additive and
opt-in: `pip install standard_quant_tools[polars]`. This tracks
[GitHub issue #1](https://github.com/karanvora2599/Standard-Tools/issues/1),
delivered in phases rather than one large cross-cutting rewrite — this
document states plainly what's supported **today** and what the roadmap
looks like, so scope is never ambiguous.

---

## What's supported today (Phase 1)

`analysis.hurst.hurst_exponent` accepts either a `pandas.Series` or a
`polars.Series` — the proof-of-concept function for this initiative,
verified to return numerically identical results for both backends
(`tests/core/test_polars_compat.py`):

```python
from standard_quant_tools.analysis.hurst import hurst_exponent
import polars as pl

result = hurst_exponent(pl.Series([...]), method="dfa")
```

`validate_series()`/`validate_dataframe()` (`standard_quant_tools.validation`)
now correctly validate a `polars.Series`/`polars.DataFrame` argument
(empty-input check, required-columns check) instead of silently skipping
validation for anything that isn't a pandas object — see "Why this
mattered" below.

## Why this mattered: the bug this phase fixed

Before this phase, `@validate_series()`/`@validate_dataframe()` checked
`isinstance(arg, pd.Series)`/`isinstance(arg, pd.DataFrame)` directly. A
caller passing a `polars.Series` wouldn't fail validation — the check
simply wouldn't match, so the decorator silently no-opped, and the
function would then fail deep inside on the first pandas-only method it
happened to call, with a confusing `AttributeError` instead of a clear
message. This is fixed at the source: both decorators now use
`standard_quant_tools._compat.is_series_like`/`is_dataframe_like`, which
check for either backend.

## The conversion boundary: `to_clean_numpy`

`standard_quant_tools._compat.to_clean_numpy(series_like, dtype=float)` is
the shared helper every dual-backend function in this library uses at its
first line — it is **not** simply `series_like.dropna().to_numpy()` for
both backends:

- `polars.Series` has no `.dropna()` at all.
- Its closest equivalent, `.drop_nulls()`, only drops `null` (missing)
  entries — Polars treats `null` and floating-point `NaN` as **distinct**
  concepts, unlike pandas' single `.dropna()`, which drops both. Matching
  pandas' actual behavior requires `.drop_nulls().drop_nans()` on the
  Polars side. `to_clean_numpy` handles this so individual functions don't
  have to think about it.

## Why the C++/numba fast paths needed zero changes

Every performance-critical kernel in this library (the pybind11 C++
extension's `hurst_dfa`/`rolling_beta`/`adx`/etc., and every numba-`@njit`
recursion added for GARCH/Kalman) already takes a raw `numpy.ndarray`, not
a pandas object directly — confirmed across every call site
(`arr = series.dropna().to_numpy(dtype=float)` immediately followed by
`_cpp.hurst_dfa(arr, ...)`, for example). Polars deliberately mirrors much
of pandas' Series API (`.to_numpy()`, `.mean()`, `.std()`, `.rolling_mean()`,
`.diff()`, `.shift()`), so most of this library's "thin wrapper over
numpy/numba/C++" functions need **conversion-boundary fixes only** —
`to_clean_numpy` plus a loosened type check — not a parallel
implementation of the underlying math.

---

## Roadmap (not yet built — tracked as follow-up work)

**Phase 2 — data-provider output conversion + the rest of the "easy tier."**
A `to_polars(df)` conversion utility for a fetched OHLCV DataFrame (via
`pl.from_pandas`) — `DataProvider.get_ohlcv`'s actual return-type contract
stays pandas (`yfinance` itself only ever produces pandas at the source,
so provider output can only ever be pandas-first, converted after the
fact). The **Polars index-becomes-a-column conversion is the explicit
boundary**: Polars has no index concept, so a DatetimeIndex becomes a real
`"Date"` column on conversion — this is a real, documented shape change,
not a transparent one. Then: `analysis.options` (no pandas/numpy at all —
likely needs nothing), `analysis.garch`, `metrics.volatility_estimators`,
`analysis.pca`, and the `indicators` rolling/ewm functions (`sma`, `ema`,
`macd`, `bollinger_bands`, `rsi`), each verified with its own dual-backend
test following `test_polars_compat.py`'s pattern.

**Phase 3 — alignment-dependent analytics, with explicit boundaries.**
`analysis.cointegration`, `analysis.regression`, `analysis.multi_factor`,
and `metrics.risk_metrics`'s `information_ratio`/`treynor_ratio` all rely
on pandas' implicit index alignment
(`common_idx = a.index.intersection(b.index); a.loc[common_idx]`) — Polars
has no index concept, so there is no direct equivalent. The plan is to
**require pre-aligned, same-length Polars input** and raise a clear
`ValidationError` naming the mismatch, rather than reimplementing pandas'
join semantics in Polars. `metrics.diagnostics`'s label-range slicing
(`.loc[start:end]`) and `.iterrows()` usage will get an explicit "Polars
not supported for this function yet" error — an honest, documented gap
rather than a risky rewrite.

**Phase 4 — not a near-term follow-up.** `backtest.engine`'s core
`run_strategy`/`_build_trade_log`/`backtest_grid` are deeply pandas-coupled
(`.shift()` for signal lag, `.cumprod()` for the equity curve, index
alignment, date-keyed dict lookups) — a genuine rewrite project, well
beyond a conversion-boundary fix. This is intentionally out of scope for
the initiative's early phases.

## What will never silently "just work"

Polars input to a function not yet covered by the phases above does NOT
yet raise a clear error. It fails the way any un-adapted pandas call
fails: `rsi(pl.Series(...))` raises `AttributeError: 'Series' object has
no attribute 'values'`, which is precisely the confusing crash this
section exists to warn about rather than a guarantee against it. Convert
explicitly with `.to_pandas()` outside the phases above — and if you hit
one that should be adapted, open an issue naming the specific function.
