#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>

#include "sqt/hurst.hpp"
#include "sqt/indicators.hpp"
#include "sqt/cointegration.hpp"
#include "sqt/backtest.hpp"
#include "sqt/rolling_regression.hpp"
#include "sqt/monte_carlo.hpp"
#include "sqt/garch.hpp"
#include "sqt/signal_state_machines.hpp"

namespace py = pybind11;

// ── Helpers ──────────────────────────────────────────────────────────────────

// Forces the input to be a C-contiguous float64 array -- but c_style and
// forcecast alone say nothing about ndim, so a caller passing a 2-D array
// (or any other shape) previously flowed through silently, flattened, and
// misinterpreted as if it were the 1-D series every kernel below assumes.
// require_1d() (called at the top of every lambda taking an Array1D
// parameter) is what actually enforces the "1-D" half of this type's name.
using Array1D = py::array_t<double, py::array::c_style | py::array::forcecast>;

static void require_1d(const Array1D& arr, const char* name) {
    if (arr.ndim() != 1)
        throw std::invalid_argument(
            std::string(name) + " must be a 1-D array, got ndim=" +
            std::to_string(arr.ndim()));
}

static py::dict hurst_result_to_dict(const sqt::HurstResult& r) {
    py::dict d;
    d["hurst"]          = r.hurst;
    d["regime"]         = r.regime;
    d["fit_r_squared"]  = r.fit_r_squared;
    d["method"]         = r.method;
    d["n_obs"]          = static_cast<py::ssize_t>(r.n_obs);
    return d;
}

// ── Module definition ─────────────────────────────────────────────────────────
//
// Every binding below follows the same GIL-release shape: extract raw
// pointers/sizes/plain-C++-value arguments from the py:: types FIRST (while
// still holding the GIL -- pybind11's array buffer access and argument
// casting are Python-API calls), then release the GIL for the duration of
// the actual sqt:: kernel call (the only part of each binding doing
// nontrivial CPU work with no Python API calls of its own), then let
// py::gil_scoped_release's destructor reacquire the GIL before touching any
// py:: type again to build the return value. This lets multiple Python
// threads run these kernels concurrently instead of serializing on the GIL
// even though NumPy released it for nothing -- the C++ call itself never
// touched a Python object once past argument extraction.

PYBIND11_MODULE(_sqt_core, m) {
    m.doc() =
        "SQT C++ core — high-performance implementations of computationally "
        "intensive functions.  Import via the public Python modules; do not "
        "call these entry-points directly.";

    // ── Hurst exponent ────────────────────────────────────────────────────────

    m.def(
        "hurst_dfa",
        [](Array1D arr, int min_window, int max_window) -> py::dict {
            require_1d(arr, "arr");
            const double* arr_ptr = arr.data();
            const auto    n       = arr.size();
            sqt::HurstResult r;
            {
                py::gil_scoped_release release;
                r = sqt::hurst_exponent(arr_ptr, n, "dfa", min_window, max_window);
            }
            return hurst_result_to_dict(r);
        },
        py::arg("arr"),
        py::arg("min_window") = 10,
        py::arg("max_window") = -1,
        "Hurst exponent via Detrended Fluctuation Analysis (DFA-1).\n\n"
        "Pass max_window=-1 to auto-select (n//4).");

    m.def(
        "hurst_rs",
        [](Array1D arr, int min_window, int max_window) -> py::dict {
            require_1d(arr, "arr");
            const double* arr_ptr = arr.data();
            const auto    n       = arr.size();
            sqt::HurstResult r;
            {
                py::gil_scoped_release release;
                r = sqt::hurst_exponent(arr_ptr, n, "rs", min_window, max_window);
            }
            return hurst_result_to_dict(r);
        },
        py::arg("arr"),
        py::arg("min_window") = 10,
        py::arg("max_window") = -1,
        "Hurst exponent via Rescaled Range (R/S) analysis.\n\n"
        "Pass max_window=-1 to auto-select (n//2).  Prefer DFA for n < 2000.");

    m.def(
        "rolling_hurst",
        [](Array1D arr, int window, int step,
           const std::string& method, int min_window) -> py::array_t<double>
        {
            require_1d(arr, "arr");
            const double* arr_ptr = arr.data();
            const auto    n       = arr.size();
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::rolling_hurst_into(arr_ptr, n, window, step, method, min_window, out_ptr);
            }
            return out;
        },
        py::arg("arr"),
        py::arg("window")     = 200,
        py::arg("step")       = 1,
        py::arg("method")     = "dfa",
        py::arg("min_window") = 10,
        "Rolling Hurst exponent in a single C++ pass — no Python re-entry per bar.\n\n"
        "Returns a 1-D float64 array of length n; first (window-1) values are NaN.");

    // ── RSI ───────────────────────────────────────────────────────────────────

    m.def(
        "rsi",
        [](Array1D arr, int period) -> py::array_t<double> {
            require_1d(arr, "arr");
            const double* arr_ptr = arr.data();
            const auto    n       = arr.size();
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::rsi_into(arr_ptr, n, period, out_ptr);
            }
            return out;
        },
        py::arg("arr"),
        py::arg("period") = 14,
        "RSI via Wilder's smoothing (SMA seed, then alpha=1/period).\n\n"
        "Returns a 1-D float64 array of length n; first `period` values are NaN.");

    // ── ADX ───────────────────────────────────────────────────────────────────

    m.def(
        "adx",
        [](Array1D high, Array1D low, Array1D close, int period) -> py::array_t<double> {
            require_1d(high, "high");
            require_1d(low, "low");
            require_1d(close, "close");
            if (high.size() != low.size() || high.size() != close.size())
                throw std::invalid_argument("high, low, close must have equal length");
            const double* high_ptr  = high.data();
            const double* low_ptr   = low.data();
            const double* close_ptr = close.data();
            const auto    n         = high.size();
            py::array_t<double> out(
                {static_cast<py::ssize_t>(n), py::ssize_t(3)});
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::adx_into(high_ptr, low_ptr, close_ptr, n, period, out_ptr);
            }
            return out;
        },
        py::arg("high"),
        py::arg("low"),
        py::arg("close"),
        py::arg("period") = 14,
        "ADX with DI+ and DI- (Wilder's smoothing).\n\n"
        "Returns a 2-D float64 array of shape (n, 3):\n"
        "  col 0 = DI+, col 1 = DI-, col 2 = ADX.\n"
        "First `period` rows have NaN in DI+/DI-; ADX starts at row 2*period-1.");

    // ── Parabolic SAR ─────────────────────────────────────────────────────────

    m.def(
        "parabolic_sar",
        [](Array1D high, Array1D low,
           double af_start, double af_step, double af_max) -> py::array_t<double>
        {
            require_1d(high, "high");
            require_1d(low, "low");
            if (high.size() != low.size())
                throw std::invalid_argument("high and low must have equal length");
            const double* high_ptr = high.data();
            const double* low_ptr  = low.data();
            const auto    n        = high.size();
            py::array_t<double> out(
                {static_cast<py::ssize_t>(n), py::ssize_t(2)});
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::parabolic_sar_into(high_ptr, low_ptr, n, af_start, af_step, af_max, out_ptr);
            }
            return out;
        },
        py::arg("high"),
        py::arg("low"),
        py::arg("af_start") = 0.02,
        py::arg("af_step")  = 0.02,
        py::arg("af_max")   = 0.2,
        "Parabolic SAR state machine.\n\n"
        "Returns a 2-D float64 array of shape (n, 2):\n"
        "  col 0 = SAR, col 1 = Trend (1.0 rising, -1.0 falling).");

    // ── Wilder's ATR ─────────────────────────────────────────────────────────

    m.def(
        "wilder_atr",
        [](Array1D high, Array1D low, Array1D close, int period) -> py::array_t<double> {
            require_1d(high, "high");
            require_1d(low, "low");
            require_1d(close, "close");
            if (high.size() != low.size() || high.size() != close.size())
                throw std::invalid_argument("high, low, close must have equal length");
            const double* high_ptr  = high.data();
            const double* low_ptr   = low.data();
            const double* close_ptr = close.data();
            const auto    n         = high.size();
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::wilder_atr_into(high_ptr, low_ptr, close_ptr, n, period, out_ptr);
            }
            return out;
        },
        py::arg("high"),
        py::arg("low"),
        py::arg("close"),
        py::arg("period") = 14,
        "Wilder's ATR (Average True Range with Wilder's smoothing).\n\n"
        "TR[0]=high[0]-low[0]; TR[i]=max(H-L,|H-C_prev|,|L-C_prev|) for i>=1.\n"
        "Seed: ATR[period-1]=mean(TR[0..period-1]).\n"
        "Forward: ATR[i]=(ATR[i-1]*(period-1)+TR[i])/period.\n\n"
        "Returns a 1-D float64 array of length n; first period-1 values are NaN.");

    // ── Backtest kernel ───────────────────────────────────────────────────────

    m.def(
        "run_strategy",
        [](Array1D prices, Array1D signals,
           double initial_capital, double commission_pct, double slippage_pct)
        -> py::dict {
            require_1d(prices, "prices");
            require_1d(signals, "signals");
            if (prices.size() != signals.size())
                throw std::invalid_argument("prices and signals must have equal length");
            const double* prices_ptr  = prices.data();
            const double* signals_ptr = signals.data();
            const auto    n           = prices.size();
            sqt::BacktestResult r;
            {
                py::gil_scoped_release release;
                r = sqt::run_strategy(
                    prices_ptr, signals_ptr, n,
                    initial_capital, commission_pct, slippage_pct);
            }

            py::array_t<double> eq(static_cast<py::ssize_t>(r.equity_curve.size()));
            std::copy(r.equity_curve.begin(), r.equity_curve.end(), eq.mutable_data());

            py::dict d;
            d["final_equity"]          = r.final_equity;
            d["total_return"]          = r.total_return;
            d["annualized_volatility"] = r.annualized_vol;
            d["sharpe_ratio"]          = r.sharpe_ratio;
            d["sortino_ratio"]         = r.sortino_ratio;
            d["max_drawdown"]          = r.max_drawdown;
            d["calmar_ratio"]          = r.calmar_ratio;
            d["num_trades"]            = r.num_trades;
            d["win_rate"]              = r.win_rate;
            d["profit_factor"]         = r.profit_factor;
            d["avg_trade_return_pct"]  = r.avg_trade_return_pct;
            d["equity_curve"]          = eq;
            return d;
        },
        py::arg("prices"),
        py::arg("signals"),
        py::arg("initial_capital") = 10'000.0,
        py::arg("commission_pct")  = 0.001,
        py::arg("slippage_pct")    = 0.0005,
        "Vectorized backtest kernel — identical algorithm to run_strategy in engine.py.\n\n"
        "One-bar lag execution: executed[i] = signals[i-1].\n"
        "Returns a dict with keys: final_equity, total_return, annualized_volatility,\n"
        "sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, num_trades,\n"
        "win_rate, profit_factor, avg_trade_return_pct, equity_curve.");

    // ── Batch backtest ────────────────────────────────────────────────────────

    m.def(
        "batch_run_strategy",
        [](Array1D prices,
           py::array_t<double, py::array::c_style | py::array::forcecast> signals_2d,
           double initial_capital, double commission_pct, double slippage_pct)
        -> py::list
        {
            require_1d(prices, "prices");
            auto prices_buf  = prices.request();
            auto signals_buf = signals_2d.request();

            if (signals_buf.ndim != 2)
                throw std::invalid_argument("signals must be a 2-D array (num_tests, n_bars)");

            const auto n         = static_cast<std::size_t>(prices_buf.shape[0]);
            const auto num_tests = static_cast<std::size_t>(signals_buf.shape[0]);

            if (static_cast<std::size_t>(signals_buf.shape[1]) != n)
                throw std::invalid_argument("signals.shape[1] must equal len(prices)");

            const double* p_ptr = static_cast<const double*>(prices_buf.ptr);
            const double* s_ptr = static_cast<const double*>(signals_buf.ptr);

            std::vector<sqt::BacktestResult> results;
            {
                py::gil_scoped_release release;
                results = sqt::batch_run_strategy(
                    p_ptr, s_ptr, n, num_tests,
                    initial_capital, commission_pct, slippage_pct);
            }

            py::list out;
            for (const auto& r : results) {
                py::dict d;
                d["final_equity"]          = r.final_equity;
                d["total_return"]          = r.total_return;
                d["annualized_volatility"] = r.annualized_vol;
                d["sharpe_ratio"]          = r.sharpe_ratio;
                d["sortino_ratio"]         = r.sortino_ratio;
                d["max_drawdown"]          = r.max_drawdown;
                d["calmar_ratio"]          = r.calmar_ratio;
                d["num_trades"]            = r.num_trades;
                d["win_rate"]              = r.win_rate;
                d["profit_factor"]         = r.profit_factor;
                d["avg_trade_return_pct"]  = r.avg_trade_return_pct;
                out.append(d);
            }
            return out;
        },
        py::arg("prices"),
        py::arg("signals"),
        py::arg("initial_capital") = 10'000.0,
        py::arg("commission_pct")  = 0.001,
        py::arg("slippage_pct")    = 0.0005,
        "Batch vectorized backtest — run all parameter combinations in one C++ call.\n\n"
        "signals must be a 2-D float64 array of shape (num_tests, n_bars).\n"
        "Returns a Python list of dicts, one per test, in input order.\n"
        "equity_curve is NOT included in the output to save memory.");

    // ── 2-variable OLS ────────────────────────────────────────────────────────

    m.def(
        "ols2",
        [](Array1D y, Array1D x) -> py::dict {
            require_1d(y, "y");
            require_1d(x, "x");
            if (y.size() != x.size())
                throw std::invalid_argument("y and x must have equal length");
            const double* y_ptr = y.data();
            const double* x_ptr = x.data();
            const auto    n     = y.size();
            sqt::Ols2Result r;
            {
                py::gil_scoped_release release;
                r = sqt::ols2(y_ptr, x_ptr, n);
            }
            py::dict d;
            d["intercept"] = r.intercept;
            d["slope"]     = r.slope;
            d["r_squared"] = r.r_squared;
            return d;
        },
        py::arg("y"),
        py::arg("x"),
        "2-variable OLS: y = intercept + slope * x.\n\n"
        "Closed-form normal equations — avoids LAPACK for this 2×2 system.\n"
        "Returns a dict with keys: intercept, slope, r_squared.");

    // ── Rolling factor loadings ───────────────────────────────────────────────

    m.def(
        "rolling_factor_loadings",
        [](Array1D y_arr,
           py::array_t<double, py::array::c_style | py::array::forcecast> factors_arr,
           int window) -> py::array_t<double>
        {
            require_1d(y_arr, "y");
            auto y_buf  = y_arr.request();
            auto f_buf  = factors_arr.request();

            if (f_buf.ndim != 2)
                throw std::invalid_argument("factors must be a 2-D array (n, k)");
            if (y_buf.shape[0] != f_buf.shape[0])
                throw std::invalid_argument("len(y) must equal factors.shape[0]");

            const auto n = static_cast<std::size_t>(y_buf.shape[0]);
            const auto k = static_cast<std::size_t>(f_buf.shape[1]);
            const int  p = static_cast<int>(k) + 1;

            const double* y_ptr = static_cast<const double*>(y_buf.ptr);
            const double* f_ptr = static_cast<const double*>(f_buf.ptr);

            py::array_t<double> out(
                {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(p)});
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::rolling_factor_loadings_into(y_ptr, f_ptr, n, k, window, out_ptr);
            }
            return out;
        },
        py::arg("y"),
        py::arg("factors"),
        py::arg("window"),
        "Rolling OLS factor loadings (incremental rank-1 updates + periodic refresh).\n\n"
        "y      : 1-D float64 array of length n (asset returns).\n"
        "factors: 2-D float64 array of shape (n, k).\n"
        "window : rolling window size in bars.\n\n"
        "Returns a 2-D float64 array of shape (n, k+1):\n"
        "  col 0 = alpha (intercept); cols 1..k = factor loadings.\n"
        "First (window-1) rows are NaN.");

    // ── Rolling beta ──────────────────────────────────────────────────────────

    m.def(
        "rolling_beta",
        [](Array1D y_arr, Array1D x_arr, int window) -> py::array_t<double>
        {
            require_1d(y_arr, "y");
            require_1d(x_arr, "x");
            if (y_arr.size() != x_arr.size())
                throw std::invalid_argument("y and x must have equal length");
            const double* y_ptr = y_arr.data();
            const double* x_ptr = x_arr.data();
            const auto    n     = static_cast<std::size_t>(y_arr.size());
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::rolling_beta_into(y_ptr, x_ptr, n, window, out_ptr);
            }
            return out;
        },
        py::arg("y"),
        py::arg("x"),
        py::arg("window"),
        "Rolling OLS beta using incremental O(1) sum updates.\n\n"
        "Returns a 1-D float64 array of length n;\n"
        "first (window-1) values are NaN.");

    // ── Bollinger Bands ───────────────────────────────────────────────────────

    m.def(
        "bollinger_bands",
        [](Array1D prices, int period, double num_std) -> py::array_t<double>
        {
            require_1d(prices, "prices");
            const double* prices_ptr = prices.data();
            const auto    n          = static_cast<std::size_t>(prices.size());
            py::array_t<double> out(
                {static_cast<py::ssize_t>(n), py::ssize_t(3)});
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::bollinger_bands_into(prices_ptr, n, period, num_std, out_ptr);
            }
            return out;
        },
        py::arg("prices"),
        py::arg("period")  = 20,
        py::arg("num_std") = 2.0,
        "Bollinger Bands — fused sliding mean+std in one pass.\n\n"
        "Returns a 2-D float64 array of shape (n, 3):\n"
        "  col 0 = Upper, col 1 = Middle (SMA), col 2 = Lower.\n"
        "First (period-1) rows are NaN.");

    // ── Stochastic Oscillator ─────────────────────────────────────────────────

    m.def(
        "stochastic_oscillator",
        [](Array1D high, Array1D low, Array1D close,
           int k_period, int d_period) -> py::array_t<double>
        {
            require_1d(high, "high");
            require_1d(low, "low");
            require_1d(close, "close");
            if (high.size() != low.size() || high.size() != close.size())
                throw std::invalid_argument("high, low, close must have equal length");
            const double* high_ptr  = high.data();
            const double* low_ptr   = low.data();
            const double* close_ptr = close.data();
            const auto    n         = static_cast<std::size_t>(high.size());
            py::array_t<double> out(
                {static_cast<py::ssize_t>(n), py::ssize_t(2)});
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::stochastic_oscillator_into(
                    high_ptr, low_ptr, close_ptr, n, k_period, d_period, out_ptr);
            }
            return out;
        },
        py::arg("high"),
        py::arg("low"),
        py::arg("close"),
        py::arg("k_period") = 14,
        py::arg("d_period") = 3,
        "Stochastic Oscillator — fused sliding min+max in one pass.\n\n"
        "Returns a 2-D float64 array of shape (n, 2):\n"
        "  col 0 = %%K, col 1 = %%D.\n"
        "First (k_period-1) rows have NaN in %%K;\n"
        "first (k_period + d_period - 2) rows have NaN in %%D.");

    // ── Engle-Granger cointegration ───────────────────────────────────────────

    m.def(
        "engle_granger",
        [](Array1D y0, Array1D y1, int max_lag, bool use_aic) -> py::dict {
            require_1d(y0, "y0");
            require_1d(y1, "y1");
            if (y0.size() != y1.size())
                throw std::invalid_argument("y0 and y1 must have equal length");
            const double* y0_ptr = y0.data();
            const double* y1_ptr = y1.data();
            const auto    n      = y0.size();
            sqt::CointResult r;
            {
                py::gil_scoped_release release;
                r = sqt::engle_granger(y0_ptr, y1_ptr, n, max_lag, use_aic);
            }
            py::dict d;
            d["intercept"]     = r.intercept;
            d["hedge_ratio"]   = r.hedge_ratio;
            d["adf_statistic"] = r.adf_statistic;
            d["optimal_lag"]   = r.optimal_lag;
            d["p_value"]       = r.p_value;
            d["cv_1pct"]       = r.cv_1pct;
            d["cv_5pct"]       = r.cv_5pct;
            d["cv_10pct"]      = r.cv_10pct;
            d["half_life"]     = r.half_life;
            d["n_obs"]         = r.n_obs;
            d["cointegrated"]  = r.cointegrated;
            return d;
        },
        py::arg("y0"),
        py::arg("y1"),
        py::arg("max_lag") = -1,
        py::arg("use_aic") = true,
        "Engle-Granger two-step cointegration test.\n\n"
        "Step 1: OLS of y0 on y1 → hedge_ratio and spread.\n"
        "Step 2: ADF test on the spread (MacKinnon 2010 critical values).\n"
        "Step 3: AR(1) half-life of the spread.\n\n"
        "Returns a dict with keys: intercept, hedge_ratio, adf_statistic,\n"
        "optimal_lag, p_value, cv_1pct, cv_5pct, cv_10pct, half_life,\n"
        "n_obs, cointegrated.");

    // ── Monte Carlo (moving-block bootstrap) ──────────────────────────────────

    m.def(
        "simulate_forward_paths",
        [](Array1D values, int horizon_days, int n_simulations, int block_size,
           double initial_capital, py::object seed) -> py::array_t<double>
        {
            require_1d(values, "values");
            const bool has_seed = !seed.is_none();
            const unsigned long long seed_val =
                has_seed ? seed.cast<unsigned long long>() : 0ULL;
             // horizon_days<=0 or n_simulations<=0 must raise, not silently
            // return a degenerate empty/zero-shaped array -- checked
            // explicitly here rather than only inferred from a result-size
            // mismatch below, since 0 * anything == 0 would otherwise make
            // an all-zero "expected" size indistinguishable from the
            // correctly-empty result these inputs actually produce.
            if (horizon_days <= 0 || n_simulations <= 0)
                throw std::invalid_argument(
                    "simulate_forward_paths: horizon_days and n_simulations must both be > 0");

            const double* values_ptr = values.data();
            const auto    n          = static_cast<std::size_t>(values.size());

            py::array_t<double> out(
                {static_cast<py::ssize_t>(n_simulations), static_cast<py::ssize_t>(horizon_days)});
            double* out_ptr = out.mutable_data();
            bool ok;
            {
                py::gil_scoped_release release;
                ok = sqt::simulate_forward_paths_into(
                    values_ptr, n, horizon_days, n_simulations, block_size,
                    initial_capital, seed_val, has_seed, out_ptr);
            }

            if (!ok)
                throw std::invalid_argument(
                    "simulate_forward_paths: invalid input (check block_size in "
                    "(0, len(values)] and initial_capital > 0)");

            return out;
        },
        py::arg("values"),
        py::arg("horizon_days"),
        py::arg("n_simulations"),
        py::arg("block_size"),
        py::arg("initial_capital"),
        py::arg("seed") = py::none(),
        "Moving-block bootstrap Monte Carlo forward simulation.\n\n"
        "Returns a 2-D float64 array of shape (n_simulations, horizon_days):\n"
        "  out[i, t] = simulated equity of path i at bar t.\n\n"
        "Each path is independently seeded (derived from `seed` and its own "
        "path index) so this does NOT reproduce numpy's PCG64 bit stream -- "
        "the same seed gives different concrete numbers than the pure-Python "
        "fallback, though repeat calls with the same seed on this path are "
        "bit-identical. Raises ValueError if horizon_days/n_simulations <= 0, "
        "values is empty, block_size is not in (0, len(values)], or "
        "initial_capital is non-positive.");

    // ── GARCH(1,1) variance recursion ─────────────────────────────────────────

    m.def(
        "garch11_variance_recursion",
        [](Array1D resid_sq, double omega, double alpha, double beta) -> py::array_t<double> {
            require_1d(resid_sq, "resid_sq");
            const double* resid_sq_ptr = resid_sq.data();
            const auto    n            = static_cast<std::size_t>(resid_sq.size());
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::garch11_variance_recursion_into(resid_sq_ptr, n, omega, alpha, beta, out_ptr);
            }
            return out;
        },
        py::arg("resid_sq"),
        py::arg("omega"),
        py::arg("alpha"),
        py::arg("beta"),
        "GARCH(1,1) conditional variance recursion.\n\n"
        "sigma2[0] = max(mean(resid_sq), 1e-12); sigma2[t] = "
        "max(omega + alpha*resid_sq[t-1] + beta*sigma2[t-1], 1e-12) for t >= 1.\n"
        "Returns a 1-D float64 array of the same length as resid_sq.");

    m.def(
        "garch11_neg_loglik",
        [](Array1D resid_sq, double omega, double alpha, double beta,
           bool penalize) -> double
        {
            require_1d(resid_sq, "resid_sq");
            const double* resid_sq_ptr = resid_sq.data();
            const auto    n            = static_cast<std::size_t>(resid_sq.size());
            double nll;
            {
                py::gil_scoped_release release;
                nll = sqt::garch11_neg_loglik(resid_sq_ptr, n, omega, alpha, beta, penalize);
            }
            return nll;
        },
        py::arg("resid_sq"),
        py::arg("omega"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("penalize") = true,
        "GARCH(1,1) negative log-likelihood -- fuses the variance recursion\n"
        "and the NLL reduction into one native call, so a scipy.optimize\n"
        "objective evaluation never round-trips a full sigma2 array across\n"
        "the Python/C++ boundary just to reduce it to a scalar.\n\n"
        "nll = 0.5 * sum(log(2*pi) + log(sigma2) + resid_sq/sigma2);\n"
        "if penalize and (alpha+beta) >= 1.0: nll += 1e6*((alpha+beta)-1)**2.\n"
        "Returns a single float (0.0 if resid_sq is empty).");

    m.def(
        "garch11_neg_loglik_grad",
        [](Array1D resid_sq, double omega, double alpha, double beta,
           bool penalize) -> py::tuple
        {
            require_1d(resid_sq, "resid_sq");
            const double* resid_sq_ptr = resid_sq.data();
            const auto    n            = static_cast<std::size_t>(resid_sq.size());
            double nll;
            double grad[3];
            {
                py::gil_scoped_release release;
                nll = sqt::garch11_neg_loglik_grad(
                    resid_sq_ptr, n, omega, alpha, beta, penalize, grad);
            }
            py::array_t<double> grad_out(3);
            std::copy(grad, grad + 3, grad_out.mutable_data());
            return py::make_tuple(nll, grad_out);
        },
        py::arg("resid_sq"),
        py::arg("omega"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("penalize") = true,
        "GARCH(1,1) negative log-likelihood AND its analytic gradient\n"
        "w.r.t. (omega, alpha, beta), computed in one fused pass -- for\n"
        "scipy.optimize's jac=True convention (fun returns (value, grad)),\n"
        "so an optimizer using the gradient pays for one recursion per\n"
        "iteration, not two.\n\n"
        "Returns a tuple (nll: float, grad: 1-D float64 array of length 3\n"
        "[d/domega, d/dalpha, d/dbeta]).");

    // ── Kalman filters (time-varying hedge ratio) ─────────────────────────────

    m.def(
        "kalman_filter_1state",
        [](Array1D y, Array1D x, double delta, double observation_noise) -> py::dict {
            require_1d(y, "y");
            require_1d(x, "x");
            if (y.size() != x.size())
                throw std::invalid_argument("y and x must have equal length");
            const double* y_ptr = y.data();
            const double* x_ptr = x.data();
            const auto    n     = static_cast<std::size_t>(y.size());
            sqt::Kalman1StateResult r;
            {
                py::gil_scoped_release release;
                r = sqt::kalman_filter_1state(y_ptr, x_ptr, n, delta, observation_noise);
            }

            py::array_t<double> beta(static_cast<py::ssize_t>(r.beta.size()));
            std::copy(r.beta.begin(), r.beta.end(), beta.mutable_data());
            py::array_t<double> gain(static_cast<py::ssize_t>(r.gain.size()));
            std::copy(r.gain.begin(), r.gain.end(), gain.mutable_data());
            py::array_t<double> innovation(static_cast<py::ssize_t>(r.innovation.size()));
            std::copy(r.innovation.begin(), r.innovation.end(), innovation.mutable_data());

            py::dict d;
            d["beta"] = beta;
            d["gain"] = gain;
            d["innovation"] = innovation;
            return d;
        },
        py::arg("y"),
        py::arg("x"),
        py::arg("delta"),
        py::arg("observation_noise"),
        "1-state (slope-only) Kalman filter for a time-varying hedge ratio.\n\n"
        "Returns a dict with keys: beta, gain, innovation (each length n).\n"
        "All-empty if delta is not in (0,1) or observation_noise <= 0.");

    m.def(
        "kalman_filter_2state",
        [](Array1D y, Array1D x, double delta, double observation_noise) -> py::dict {
            require_1d(y, "y");
            require_1d(x, "x");
            if (y.size() != x.size())
                throw std::invalid_argument("y and x must have equal length");
            const double* y_ptr = y.data();
            const double* x_ptr = x.data();
            const auto    n     = static_cast<std::size_t>(y.size());
            sqt::Kalman2StateResult r;
            {
                py::gil_scoped_release release;
                r = sqt::kalman_filter_2state(y_ptr, x_ptr, n, delta, observation_noise);
            }

            py::array_t<double> alpha(static_cast<py::ssize_t>(r.alpha.size()));
            std::copy(r.alpha.begin(), r.alpha.end(), alpha.mutable_data());
            py::array_t<double> beta(static_cast<py::ssize_t>(r.beta.size()));
            std::copy(r.beta.begin(), r.beta.end(), beta.mutable_data());
            py::array_t<double> gain(static_cast<py::ssize_t>(r.gain.size()));
            std::copy(r.gain.begin(), r.gain.end(), gain.mutable_data());
            py::array_t<double> innovation(static_cast<py::ssize_t>(r.innovation.size()));
            std::copy(r.innovation.begin(), r.innovation.end(), innovation.mutable_data());

            py::dict d;
            d["alpha"] = alpha;
            d["beta"] = beta;
            d["gain"] = gain;
            d["innovation"] = innovation;
            return d;
        },
        py::arg("y"),
        py::arg("x"),
        py::arg("delta"),
        py::arg("observation_noise"),
        "2-state (intercept + slope) Kalman filter for a time-varying hedge ratio.\n\n"
        "Returns a dict with keys: alpha, beta, gain, innovation (each length n).\n"
        "All-empty if delta is not in (0,1) or observation_noise <= 0.");

    // ── Signal state machines (Donchian / VWAP-reversion hysteresis) ──────────

    m.def(
        "donchian_state_machine",
        [](Array1D close, Array1D entry_max, Array1D exit_min) -> py::array_t<double> {
            require_1d(close, "close");
            require_1d(entry_max, "entry_max");
            require_1d(exit_min, "exit_min");
            if (close.size() != entry_max.size() || close.size() != exit_min.size())
                throw std::invalid_argument("close, entry_max, exit_min must have equal length");
            const double* close_ptr     = close.data();
            const double* entry_max_ptr = entry_max.data();
            const double* exit_min_ptr  = exit_min.data();
            const auto    n             = static_cast<std::size_t>(close.size());
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::donchian_state_machine_into(
                    close_ptr, entry_max_ptr, exit_min_ptr, n, out_ptr);
            }
            return out;
        },
        py::arg("close"),
        py::arg("entry_max"),
        py::arg("exit_min"),
        "Donchian breakout entry/exit hysteresis: 1.0=long, 0.0=flat.\n\n"
        "A NaN in entry_max/exit_min (rolling warmup) does not update the\n"
        "position state for that bar; output carries the position already\n"
        "held instead of hardcoding 0.0.");

    m.def(
        "vwap_reversion_state_machine",
        [](Array1D close, Array1D vwap, double entry_threshold) -> py::array_t<double> {
            require_1d(close, "close");
            require_1d(vwap, "vwap");
            if (close.size() != vwap.size())
                throw std::invalid_argument("close and vwap must have equal length");
            const double* close_ptr = close.data();
            const double* vwap_ptr  = vwap.data();
            const auto    n         = static_cast<std::size_t>(close.size());
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            double* out_ptr = out.mutable_data();
            {
                py::gil_scoped_release release;
                sqt::vwap_reversion_state_machine_into(
                    close_ptr, vwap_ptr, entry_threshold, n, out_ptr);
            }
            return out;
        },
        py::arg("close"),
        py::arg("vwap"),
        py::arg("entry_threshold"),
        "VWAP mean-reversion entry/exit hysteresis: 1.0=long, 0.0=flat.\n\n"
        "A NaN in vwap (rolling warmup) does not update the position state\n"
        "for that bar; output carries the position already held instead of\n"
        "hardcoding 0.0.");
}
