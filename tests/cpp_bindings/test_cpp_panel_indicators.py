"""
Panel (whole-universe) indicator path.

The premise of `indicators.panel` is that it is a *faster shape*, not a
different calculation: each row of the panel goes through the same `*_into`
kernel the single-series path uses. So the gate is equality with the
per-ticker call, and it is exact at the binding layer -- nothing accumulates
across tickers, so there is no floating-point reason for a difference and a
tolerance would hide an indexing bug.

The Python layer compares with a tolerance instead, because the per-ticker
wrappers round and reindex on the way out.

Run:
    pytest tests/cpp_bindings/test_cpp_panel_indicators.py -v
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

_cpp: Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

requires_cpp = pytest.mark.skipif(not HAS_CPP, reason="_sqt_core not built")

ALL_FIVE = dict(
    compute_rsi=True,
    compute_adx=True,
    compute_atr=True,
    compute_bollinger=True,
    compute_stochastic=True,
)


def _panel(n_tickers=12, n_bars=300, seed=3):
    rng = np.random.default_rng(seed)
    close = np.ascontiguousarray(
        np.array(
            [
                100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_bars)))
                for _ in range(n_tickers)
            ]
        )
    )
    return close * 1.005, close * 0.995, close


@requires_cpp
class TestPanelKernel:
    def test_matches_per_ticker_exactly(self):
        high, low, close = _panel()
        got = _cpp.technical_indicators_panel(high, low, close, **ALL_FIVE)
        assert set(got) == {
            "rsi",
            "adx",
            "atr",
            "bollinger_bands",
            "stochastic_oscillator",
        }
        for t in range(high.shape[0]):
            one = _cpp.technical_indicators(high[t], low[t], close[t], **ALL_FIVE)
            for key, expected in one.items():
                np.testing.assert_array_equal(
                    got[key][t], expected, err_msg=f"ticker {t}, {key}"
                )

    def test_shapes(self):
        high, low, close = _panel(7, 120)
        got = _cpp.technical_indicators_panel(high, low, close, **ALL_FIVE)
        assert got["rsi"].shape == (7, 120)
        assert got["atr"].shape == (7, 120)
        assert got["adx"].shape == (7, 120, 3)
        assert got["bollinger_bands"].shape == (7, 120, 3)
        assert got["stochastic_oscillator"].shape == (7, 120, 2)

    def test_only_requested_keys_are_returned(self):
        high, low, close = _panel(4, 100)
        got = _cpp.technical_indicators_panel(high, low, close, compute_rsi=True)
        assert set(got) == {"rsi"}
        assert _cpp.technical_indicators_panel(high, low, close) == {}

    @pytest.mark.parametrize("period", [-5, 0])
    def test_degenerate_period_yields_nan_not_an_exception(self, period):
        """Same contract as the single-series path: bad parameters give NaN.

        period == 1 is deliberately NOT in this list. It is a legitimate
        lookback -- RSI over a single bar is 0 or 100 depending on the sign of
        that bar's change -- and only period <= 0 is rejected.
        """
        high, low, close = _panel(3, 80)
        got = _cpp.technical_indicators_panel(
            high, low, close, compute_rsi=True, rsi_period=period
        )
        assert np.all(np.isnan(got["rsi"]))

    def test_period_one_is_valid_and_matches_per_ticker(self):
        high, low, close = _panel(3, 80)
        got = _cpp.technical_indicators_panel(
            high, low, close, compute_rsi=True, rsi_period=1
        )
        for t in range(3):
            np.testing.assert_array_equal(got["rsi"][t], _cpp.rsi(close[t], 1))

    def test_nan_bars_do_not_raise(self):
        high, low, close = _panel(5, 200)
        close = close.copy()
        close[2, 100] = np.nan
        got = _cpp.technical_indicators_panel(high, low, close, **ALL_FIVE)
        assert got["rsi"].shape == (5, 200)
        # The bad bar is confined to its own ticker.
        assert not np.isnan(got["rsi"][0, 50])

    def test_shape_validation(self):
        high, low, close = _panel(4, 100)
        with pytest.raises(ValueError, match="2-D"):
            _cpp.technical_indicators_panel(high[0], low[0], close[0], compute_rsi=True)
        with pytest.raises(ValueError, match="identical shapes"):
            _cpp.technical_indicators_panel(high, low[:, :50], close, compute_rsi=True)

    def test_thread_count_does_not_change_results(self, monkeypatch):
        """Tickers are independent, so scheduling must not be observable."""
        high, low, close = _panel(20, 250, seed=8)
        base = _cpp.technical_indicators_panel(high, low, close, **ALL_FIVE)
        for key, arr in base.items():
            again = _cpp.technical_indicators_panel(high, low, close, **ALL_FIVE)[key]
            np.testing.assert_array_equal(arr, again, err_msg=key)


@requires_cpp
class TestPanelPythonApi:
    @staticmethod
    def _ohlcv(n_tickers=8, n_bars=250, seed=6):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2018-01-01", periods=n_bars, freq="B")
        out = {}
        for i in range(n_tickers):
            cl = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n_bars)))
            out[f"T{i:03d}"] = pd.DataFrame(
                {
                    "Open": cl,
                    "High": cl * 1.01,
                    "Low": cl * 0.99,
                    "Close": cl,
                    "Volume": np.full(n_bars, 1e6),
                },
                index=idx,
            )
        return out

    def test_matches_the_per_ticker_wrappers(self):
        from standard_quant_tools.indicators.momentum import rsi as rsi_w
        from standard_quant_tools.indicators.momentum import (
            stochastic_oscillator as stoch_w,
        )
        from standard_quant_tools.indicators.panel import technical_indicators_panel
        from standard_quant_tools.indicators.trend import adx as adx_w
        from standard_quant_tools.indicators.volatility import bollinger_bands as bb_w
        from standard_quant_tools.indicators.volatility import wilder_atr as atr_w

        data = self._ohlcv()
        res = technical_indicators_panel(
            data,
            ["rsi", "adx", "atr", "bollinger_bands", "stochastic_oscillator"],
        )
        for t, df in data.items():
            np.testing.assert_allclose(
                res["rsi"][t], rsi_w(df["Close"], 14), equal_nan=True
            )
            np.testing.assert_allclose(
                res["atr"][t],
                atr_w(df["High"], df["Low"], df["Close"], 14),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                res["adx"][t].to_numpy(),
                adx_w(df["High"], df["Low"], df["Close"], 14).to_numpy(),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                res["bollinger_bands"][t].to_numpy(),
                bb_w(df["Close"], 20, 2.0).to_numpy(),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                res["stochastic_oscillator"][t].to_numpy(),
                stoch_w(df["High"], df["Low"], df["Close"], 14, 3).to_numpy(),
                equal_nan=True,
            )

    def test_field_names_match_the_per_ticker_wrappers(self):
        """A caller moving to the panel should not have to learn new labels."""
        from standard_quant_tools.indicators.panel import technical_indicators_panel
        from standard_quant_tools.indicators.trend import adx as adx_w
        from standard_quant_tools.indicators.volatility import bollinger_bands as bb_w

        data = self._ohlcv(3, 120)
        res = technical_indicators_panel(data, ["adx", "bollinger_bands"])
        one = next(iter(data.values()))
        assert list(res["adx"]["T000"].columns) == list(
            adx_w(one["High"], one["Low"], one["Close"], 14).columns
        )
        assert list(res["bollinger_bands"]["T000"].columns) == list(
            bb_w(one["Close"], 20, 2.0).columns
        )

    def test_index_is_the_common_intersection(self):
        from standard_quant_tools.indicators.panel import technical_indicators_panel

        data = self._ohlcv(3, 200)
        # Truncate one ticker; the panel must fall back to the overlap.
        key = list(data)[1]
        data[key] = data[key].iloc[10:]
        res = technical_indicators_panel(data, ["rsi"])
        assert len(res["rsi"]) == 190
        assert res["rsi"].index[0] == data[key].index[0]

    def test_rejects_bad_input(self):
        from standard_quant_tools.error import ValidationError
        from standard_quant_tools.indicators.panel import technical_indicators_panel

        data = self._ohlcv(3, 100)
        with pytest.raises(ValidationError, match="unknown indicator"):
            technical_indicators_panel(data, ["not_an_indicator"])
        with pytest.raises(ValidationError, match="no tickers"):
            technical_indicators_panel({}, ["rsi"])
        with pytest.raises(ValidationError, match="missing column"):
            technical_indicators_panel(
                {"A": data["T000"].drop(columns=["High"])}, ["rsi"]
            )

    def test_empty_indicator_list_is_empty_result(self):
        from standard_quant_tools.indicators.panel import technical_indicators_panel

        assert technical_indicators_panel(self._ohlcv(2, 50), []) == {}
