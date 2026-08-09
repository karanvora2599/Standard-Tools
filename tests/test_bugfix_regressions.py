"""
Regression tests for the bug-fix pass documented in CHANGELOG.

Each test here pins a specific defect that was found by auditing the Python
codebase, so a future refactor can't silently reintroduce it. Grouped by the
module the defect lived in, with the failure mode spelled out in each
docstring — several of these were silent-wrong-answer bugs rather than
crashes, which is exactly the kind that comes back unnoticed.
"""

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.cointegration import (
    cointegration_test,
    kalman_hedge_ratio,
)
from standard_quant_tools.analysis.multi_factor import (
    multi_factor_regression,
    rolling_factor_loadings,
)
from standard_quant_tools.audit.hashing import hash_dataframe, hash_payload
from standard_quant_tools.backtest.costs import per_share_commission
from standard_quant_tools.backtest.engine import (
    _build_trade_log,
    _compute_trade_stats,
    run_strategy,
)
from standard_quant_tools.backtest.panel import run_signal_panel_backtest
from standard_quant_tools.backtest.robustness import parameter_sensitivity
from standard_quant_tools.data._cache import _parquet_path
from standard_quant_tools.data._retry import retry
from standard_quant_tools.error import (
    APIError,
    NonRetryableAPIError,
    ValidationError,
)
from standard_quant_tools.indicators.momentum import stochastic_oscillator
from standard_quant_tools.indicators.trend import _adx_numba, _psar_numba, adx, macd
from standard_quant_tools.indicators.volatility import atr
from standard_quant_tools.indicators.volume import mfi
from standard_quant_tools.metrics.return_metrics import cagr
from standard_quant_tools.metrics.risk_metrics import evt_tail_risk


def _ohlcv(n=30, start="2024-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    base = np.linspace(100.0, 100.0 + n - 1, n)
    return pd.DataFrame(
        {
            "Open": base,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base + 0.5,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


# ── indicators/trend.py: numba out-of-bounds writes ──────────────────────────


class TestNumbaBoundsSafety:
    """
    _adx_numba wrote result[period], dx_vals[period] and result[2*period-1]
    without ever checking them against n. @njit compiles with bounds checking
    DISABLED, so with n <= period those were out-of-bounds heap writes that
    returned "successfully" instead of raising — while the same code run as
    pure Python (numba absent) raised IndexError and the C++ kernel returned
    all-NaN. Three paths, three behaviors, one of them memory-unsafe.
    """

    @pytest.mark.parametrize("period", [14, 20, 50])
    def test_adx_numba_kernel_short_input_returns_all_nan(self, period):
        n = 10
        h = np.linspace(101, 110, n)
        low = np.linspace(99, 108, n)
        c = np.linspace(100, 109, n)
        out = _adx_numba(h, low, c, period)
        assert out.shape == (n, 3)
        assert np.isnan(out).all()

    def test_adx_numba_pure_python_agrees_with_jit(self):
        """The un-jitted reference must not raise IndexError either."""
        n = 10
        h = np.linspace(101, 110, n)
        low = np.linspace(99, 108, n)
        c = np.linspace(100, 109, n)
        py_func = getattr(_adx_numba, "py_func", _adx_numba)
        assert np.isnan(py_func(h, low, c, 14)).all()

    def test_adx_public_api_short_input_all_nan(self):
        df = _ohlcv(n=10)
        out = adx(df["High"], df["Low"], df["Close"], period=14)
        assert out.isna().all().all()

    def test_psar_numba_empty_input(self):
        empty = np.array([], dtype=float)
        assert _psar_numba(empty, empty, 0.02, 0.02, 0.2).shape == (0, 2)
        py_func = getattr(_psar_numba, "py_func", _psar_numba)
        assert py_func(empty, empty, 0.02, 0.02, 0.2).shape == (0, 2)

    def test_adx_rejects_mismatched_lengths(self):
        df = _ohlcv(n=30)
        with pytest.raises(ValidationError, match="same length"):
            adx(df["High"], df["Low"].iloc[:10], df["Close"])


# ── data/_retry.py ───────────────────────────────────────────────────────────


class TestRetrySemantics:
    def test_validation_error_keeps_its_type(self):
        """
        The broad `except Exception` used to re-wrap every non-listed
        exception as APIError, so a caller's `except ValidationError` never
        fired for a bad-argument error raised inside a decorated provider
        call.
        """

        @retry(times=3, delay=0)
        def f():
            raise ValidationError("bad period")

        with pytest.raises(ValidationError):
            f()

    def test_times_zero_is_rejected(self):
        """retry(times=0) silently returned None WITHOUT calling the function."""
        with pytest.raises(ValueError, match="must be >= 1"):
            retry(times=0)

    def test_transient_network_error_is_retried(self):
        """
        ConnectionError/TimeoutError are neither ValueError nor APIError, so
        they hit the catch-all and were raised on the FIRST attempt — the
        single most common transient failure went unretried.
        """
        calls = []

        @retry(times=3, delay=0)
        def f():
            calls.append(1)
            raise ConnectionError("network down")

        with pytest.raises(APIError):
            f()
        assert len(calls) == 3

    def test_transient_error_recovers(self):
        calls = []

        @retry(times=3, delay=0)
        def f():
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError("slow")
            return "ok"

        assert f() == "ok"
        assert len(calls) == 3

    def test_api_error_still_retried(self):
        calls = []

        @retry(times=3, delay=0)
        def f():
            calls.append(1)
            raise APIError("429 rate limited")

        with pytest.raises(APIError):
            f()
        assert len(calls) == 3

    def test_non_retryable_api_error_not_retried(self):
        calls = []

        @retry(times=3, delay=0)
        def f():
            calls.append(1)
            raise NonRetryableAPIError("401 bad key")

        with pytest.raises(NonRetryableAPIError):
            f()
        assert len(calls) == 1


# ── backtest/engine.py: finite-input contract ────────────────────────────────


class TestEngineInputContract:
    """
    require_finite_array ran only inside the C++ branch, so the SAME call with
    the SAME data raised with _sqt_core built and silently produced NaN
    metrics without it. next_open/hl2_exploratory were never checked at all —
    and there a NaN reference price is worse than NaN-poisoning, because
    pandas' cumprod is skipna=True: the bar's return is DROPPED and
    total_return is computed over a quietly shortened series.
    """

    def test_nan_open_rejected_for_next_open(self):
        df = _ohlcv()
        df.loc[df.index[4], "Open"] = np.nan
        sig = pd.Series(1.0, index=df.index)
        with pytest.raises(ValidationError, match="Open"):
            run_strategy(df, sig, fill_price="next_open")

    @pytest.mark.parametrize("col", ["High", "Low"])
    def test_nan_hl_rejected_for_hl2(self, col):
        df = _ohlcv()
        df.loc[df.index[3], col] = np.nan
        sig = pd.Series(1.0, index=df.index)
        with pytest.raises(ValidationError, match=col):
            run_strategy(df, sig, fill_price="hl2_exploratory")

    def test_nan_signal_rejected_on_every_path(self):
        df = _ohlcv()
        sig = pd.Series(1.0, index=df.index)
        sig.iloc[5] = np.nan
        with pytest.raises(ValidationError, match="signals"):
            run_strategy(df, sig, fill_price="next_open")

    def test_missing_open_column_named_clearly(self):
        df = _ohlcv()[["Close"]]
        sig = pd.Series(1.0, index=df.index)
        with pytest.raises(ValidationError, match="Open"):
            run_strategy(df, sig, fill_price="next_open")

    def test_clean_next_open_has_no_nan_hole(self):
        df = _ohlcv()
        sig = pd.Series(1.0, index=df.index)
        result = run_strategy(df, sig, fill_price="next_open")
        assert int(result["equity_curve"].isna().sum()) == 0


# ── metrics: EVT, CAGR ───────────────────────────────────────────────────────


class TestEvtTailRisk:
    @staticmethod
    def _returns():
        rng = np.random.default_rng(0)
        return pd.Series(rng.standard_t(4, 3000) / 100.0)

    def test_confidence_below_threshold_rejected(self):
        """
        With confidence <= 1 - tail_fraction the exceedance probability is
        >= 1, so the POT formula returned a "VaR" BELOW its own threshold —
        a silently wrong number rather than a less precise one.
        """
        with pytest.raises(ValidationError, match="1 - tail_fraction"):
            evt_tail_risk(self._returns(), confidence=0.90, tail_fraction=0.05)

    def test_valid_confidence_var_exceeds_threshold(self):
        out = evt_tail_risk(self._returns(), confidence=0.99, tail_fraction=0.05)
        assert out["var_evt"] >= out["threshold"]


class TestCagrNonPositiveTerminal:
    def test_wiped_out_equity_reports_total_loss_not_nan(self):
        """
        (1 + total_ret) ** (1/years) with total_ret <= -1 yields NaN plus a
        RuntimeWarning, which then propagated silently into calmar_ratio.
        """
        eq = pd.Series([10000.0, 8000.0, 3000.0, -500.0, -800.0])
        out = cagr(eq)
        assert not np.isnan(out)
        assert out == -1.0


# ── backtest/robustness.py ───────────────────────────────────────────────────


class TestParameterSensitivityNaN:
    def test_nan_metric_cannot_become_best(self):
        """
        np.sort puts NaN last, so [::-1] put it FIRST — one NaN Sharpe (a grid
        row with zero-variance returns is the common source) became `best` and
        made every reported gap NaN.
        """
        grid = pd.DataFrame(
            {"sharpe_ratio": [1.5, np.nan, 0.9, 0.4], "p": [1, 2, 3, 4]}
        )
        out = parameter_sensitivity(grid)
        assert out["best"] == 1.5
        assert out["n_trials"] == 3
        assert not np.isnan(out["best_minus_median"])

    def test_all_nan_metric_raises(self):
        grid = pd.DataFrame({"sharpe_ratio": [np.nan, np.nan]})
        with pytest.raises(ValidationError, match="no finite values"):
            parameter_sensitivity(grid)


# ── audit/hashing.py ─────────────────────────────────────────────────────────


class TestAuditHashCollisions:
    def test_column_names_are_part_of_the_fingerprint(self):
        """
        pd.util.hash_pandas_object is a per-row digest that never sees column
        labels, so two frames of identical numbers under entirely different
        column names produced the same provenance hash.
        """
        a = pd.DataFrame({"Close": [1.0, 2.0], "Open": [3.0, 4.0]})
        b = pd.DataFrame({"Volume": [1.0, 2.0], "Adj": [3.0, 4.0]})
        assert hash_dataframe(a) != hash_dataframe(b)

    def test_column_order_changes_fingerprint(self):
        a = pd.DataFrame({"Close": [1.0, 2.0], "Open": [3.0, 4.0]})
        assert hash_dataframe(a) != hash_dataframe(a[["Open", "Close"]])

    def test_values_still_detected_and_deterministic(self):
        a = pd.DataFrame({"Close": [1.0, 2.0], "Open": [3.0, 4.0]})
        assert hash_dataframe(a) == hash_dataframe(a.copy())
        mutated = a.copy()
        mutated.loc[0, "Close"] = 99.0
        assert hash_dataframe(a) != hash_dataframe(mutated)

    def test_large_ndarray_not_truncated(self):
        """
        default=str routed ndarrays through numpy's abbreviating repr, so two
        large arrays differing only in the middle hashed identically.
        """
        x = np.arange(10_000)
        y = x.copy()
        y[5_000] = -1
        assert hash_payload({"o": x}) != hash_payload({"o": y})

    def test_chain_hash_unchanged_for_plain_json_records(self):
        """
        hash_payload builds the tamper-evident record chain, so its output for
        ordinary JSON-typed records must be byte-identical to the pre-fix
        implementation or every existing audit trail stops verifying.
        """
        record = {
            "request_id": "abc",
            "tool_name": "t",
            "duration_ms": 1.5,
            "input": {"a": [1, 2, 3]},
            "record_hash": None,
            "ok": True,
            "n": None,
        }
        legacy = hashlib.sha256(
            json.dumps(record, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        assert hash_payload(record) == legacy


# ── cross-path consistency ───────────────────────────────────────────────────


class TestPathConsistency:
    def test_stochastic_flat_window_matches_cpp_convention(self):
        """
        A zero-range window made %K a 0/0: the C++ kernel returned 0.0 and the
        pandas fallback returned NaN, so the answer depended only on whether
        _sqt_core happened to be built.
        """
        flat = pd.Series([100.0] * 20, index=pd.date_range("2024-01-01", periods=20))
        out = stochastic_oscillator(flat, flat, flat)
        assert (out["Stoch_K"].dropna() == 0.0).all()
        # Warm-up must still be NaN, not 0.0.
        assert bool(np.isnan(out["Stoch_K"].iloc[0]))

    def test_cointegration_rejects_unknown_autolag(self):
        """
        The C++ path mapped anything != "bic" onto AIC while the statsmodels
        fallback passed the string through to coint(), so a typo ran a
        different criterion depending on the build.
        """
        idx = pd.date_range("2020-01-01", periods=200, freq="D")
        rng = np.random.default_rng(1)
        b = pd.Series(np.cumsum(rng.normal(0, 1, 200)) + 100, index=idx)
        a = b * 1.5 + rng.normal(0, 0.5, 200)
        with pytest.raises(ValidationError, match="autolag"):
            cointegration_test(a, b, autolag="t-stat")


# ── validation-consistency gaps ──────────────────────────────────────────────


class TestFiniteInputConsistency:
    def test_atr_rejects_nan(self):
        df = _ohlcv()
        bad = df["High"].copy()
        bad.iloc[5] = np.nan
        with pytest.raises(ValidationError, match="non-finite"):
            atr(bad, df["Low"], df["Close"])

    def test_kalman_hedge_ratio_rejects_nan(self):
        idx = pd.date_range("2024-01-01", periods=50)
        a = pd.Series(np.linspace(100, 150, 50), index=idx)
        b = a.copy()
        b.iloc[10] = np.nan
        with pytest.raises(ValidationError, match="non-finite"):
            kalman_hedge_ratio(a, b)

    def test_multi_factor_regression_rejects_nan(self):
        idx = pd.date_range("2024-01-01", periods=50)
        rng = np.random.default_rng(2)
        y = pd.Series(rng.normal(0, 0.01, 50), index=idx)
        f = pd.DataFrame({"mkt": rng.normal(0, 0.01, 50)}, index=idx)
        f.iloc[7, 0] = np.nan
        with pytest.raises(ValidationError, match="non-finite"):
            multi_factor_regression(y, f)


class TestCppPythonEquivalence:
    """
    Cross-path equivalence for the C++ audit's two confirmed divergences.

    These assert the two backends against EACH OTHER rather than against a
    constant — the failure mode in both cases was "same call, different answer
    depending on whether _sqt_core happened to be built", which a test pinning
    only one side cannot see.
    """

    @staticmethod
    def _cpp():
        try:
            from standard_quant_tools import _sqt_core

            return _sqt_core
        except ImportError:
            return None

    def test_profit_factor_zero_over_zero_agrees(self):
        """
        Flat prices + zero costs → every trade returns exactly 0.0, so
        gross_win and gross_loss are both 0. C++ returned 0.0 here while
        Python returned inf; both must now report inf ("no losing trades").
        """
        cpp = self._cpp()
        if cpp is None:
            pytest.skip("_sqt_core not built")

        n = 10
        prices = np.full(n, 100.0)
        signals = np.full(n, 1.0)

        cpp_pf = cpp.run_strategy(prices, signals, 10_000.0, 0.0, 0.0)["profit_factor"]

        idx = pd.RangeIndex(n)
        p = pd.Series(prices, index=idx)
        executed = pd.Series(signals, index=idx).shift(1).fillna(0.0)
        py_pf = _compute_trade_stats(_build_trade_log(p.shift(1), p, executed, 0.0))[
            "profit_factor"
        ]

        assert math.isinf(cpp_pf), f"C++ profit_factor should be inf, got {cpp_pf}"
        assert math.isinf(py_pf), f"Python profit_factor should be inf, got {py_pf}"

    def test_rolling_factor_loadings_underdetermined_agrees(self):
        """
        window < k+2: the C++ kernel returns all-NaN; the Python fallback used
        to return numpy's minimum-norm lstsq solution instead. Both must now
        be all-NaN.
        """
        n = 40
        idx = pd.date_range("2024-01-01", periods=n)
        rng = np.random.default_rng(3)
        y = pd.Series(rng.normal(0, 0.01, n), index=idx)
        f = pd.DataFrame({"f": rng.normal(0, 0.01, n)}, index=idx)

        for window in (1, 2):  # k=1 → k+2 == 3
            out = rolling_factor_loadings(y, f, window=window)
            assert out.isna().all().all(), f"window={window} must be all-NaN"

        cpp = self._cpp()
        if cpp is not None:
            arr = np.ascontiguousarray(f.to_numpy(dtype=np.float64))
            raw = cpp.rolling_factor_loadings(y.to_numpy(dtype=np.float64), arr, 2)
            assert np.isnan(raw).all()


class TestMiscEdgeCases:
    def test_zero_share_trade_costs_nothing(self):
        """The minimum is a per-ORDER floor; no order means no commission."""
        assert per_share_commission(0.0, 0.005, minimum=1.0) == 0.0
        assert per_share_commission(10.0, 0.005, minimum=1.0) == 1.0

    def test_slash_and_dash_symbols_get_distinct_cache_paths(self):
        a = _parquet_path("BRK/B", "2022-01-01", "2023-01-01", "1d")
        b = _parquet_path("BRK-B", "2022-01-01", "2023-01-01", "1d")
        assert a != b

    def test_macd_rejects_inverted_periods(self):
        s = pd.Series(np.linspace(100, 200, 60))
        with pytest.raises(ValidationError, match="must be <"):
            macd(s, fast=26, slow=12)

    def test_mfi_no_flow_window_is_nan_not_zero(self):
        """
        With both pos_flow and neg_flow zero there is no money flow at all, so
        MFI is undefined. The second unconditional .where() used to overwrite
        the first one's 100.0 with 0.0, reporting "maximally oversold".
        """
        n = 30
        idx = pd.date_range("2024-01-01", periods=n)
        flat = pd.Series(100.0, index=idx)
        zero_vol = pd.Series(0.0, index=idx)
        out = mfi(flat, flat, flat, zero_vol, period=14)
        assert out.iloc[-1] != 0.0
        assert bool(np.isnan(out.iloc[-1]))

    def test_panel_weights_are_validated(self):
        df = _ohlcv(n=60)
        price_data = {"AAA": df, "BBB": df}
        panel = pd.DataFrame({"AAA": 1.0, "BBB": 1.0}, index=df.index, dtype=float)
        with pytest.raises(ValidationError, match="sum to 1.0"):
            run_signal_panel_backtest(price_data, panel, weights=[0.3, 0.3])
        with pytest.raises(ValidationError, match="missing entries"):
            run_signal_panel_backtest(price_data, panel, weights={"AAA": 1.0})
        with pytest.raises(ValidationError, match="length"):
            run_signal_panel_backtest(price_data, panel, weights=[1.0])
